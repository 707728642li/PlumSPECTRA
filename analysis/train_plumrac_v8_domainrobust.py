from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4
import train_plumrac_v5_auxpretrain as v5


DOMAIN_OBJECTIVE = "group_dro"
ROBUST_WEIGHT = 0.35
GROUP_DRO_TEMPERATURE = 0.20
VREX_WEIGHT = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    element = F.smooth_l1_loss(prediction, target, beta=0.5, reduction="none")
    group_losses = torch.stack([element[group == value].mean() for value in torch.unique(group)])
    mean_group_loss = group_losses.mean()
    if DOMAIN_OBJECTIVE == "group_dro":
        temperature = max(GROUP_DRO_TEMPERATURE, 1e-6)
        robust = temperature * (
            torch.logsumexp(group_losses / temperature, dim=0)
            - math.log(float(len(group_losses)))
        )
        objective = (1.0 - ROBUST_WEIGHT) * mean_group_loss + ROBUST_WEIGHT * robust
    elif DOMAIN_OBJECTIVE == "vrex":
        normalized_variance = group_losses.var(unbiased=False) / mean_group_loss.detach().clamp_min(1e-4)
        objective = mean_group_loss + VREX_WEIGHT * normalized_variance
    else:
        raise ValueError(DOMAIN_OBJECTIVE)
    return objective, group_losses.detach()


def train_domain_robust_residual_model(
    raw: np.ndarray,
    wavelength: np.ndarray,
    clean_channels: np.ndarray,
    residual_target: np.ndarray,
    anchor_standardized: np.ndarray,
    group_index: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    seed: int,
    config: v2.RACConfig,
    device: torch.device,
    fixed_epochs: int | None = None,
):
    v2.set_seed(seed)
    channel_mean, channel_sd = v2.fit_channel_scaler(clean_channels, train_indices)
    channel_builder = v2.SpectralChannelBuilder(wavelength, channel_mean, channel_sd, config).to(device)
    encoder_state, pretrain_history = v5.pretrain_encoder(
        raw,
        channel_builder,
        groups,
        train_indices,
        seed,
        config,
        device,
    )
    model = v4.V4PlumRACNet(config.width, config.blocks, config.dropout, config.attention_tail).to(device)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if unexpected or any(not key.startswith("trait_tail") for key in missing):
        raise RuntimeError(f"Invalid auxiliary encoder transfer: missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = fixed_epochs if fixed_epochs is not None else config.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sampler = v2.cultivar_balanced_sampler(groups, train_indices, config.sampler_power, seed)
    train_loader = DataLoader(
        v2.SpectrumDataset(raw, residual_target, anchor_standardized, group_index, train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = None
    if validation_indices is not None:
        validation_loader = DataLoader(
            v2.SpectrumDataset(raw, residual_target, anchor_standardized, group_index, validation_indices),
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )

    best_score = math.inf
    best_epoch = epochs
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        spreads: list[float] = []
        for raw_batch, residual_batch, anchor_batch, group_batch in train_loader:
            raw_batch = raw_batch.to(device, non_blocking=True)
            residual_batch = residual_batch.to(device, non_blocking=True)
            anchor_batch = anchor_batch.to(device, non_blocking=True)
            group_batch = group_batch.to(device, non_blocking=True)
            channels = channel_builder(raw_batch, augment=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(channels, anchor_batch)
                regression, group_losses = domain_regression_loss(prediction, residual_batch, group_batch)
                if config.structure_on_final:
                    final_prediction = anchor_batch + prediction
                    final_target = anchor_batch + residual_batch
                else:
                    final_prediction = prediction
                    final_target = residual_batch
                centred = v2.centred_batch_loss(final_prediction, final_target, group_batch)
                ranking = v2.pairwise_rank_loss(
                    final_prediction,
                    final_target,
                    group_batch,
                    config.rank_temperature,
                )
                loss = regression + config.center_weight * centred + config.rank_weight * ranking
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            spreads.append(float((group_losses.max() - group_losses.min()).cpu()))
        scheduler.step()
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "train_group_loss_spread": float(np.mean(spreads)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "auxiliary_pretrain_final_loss": float(pretrain_history[-1]),
        }
        if validation_loader is not None:
            validation_prediction, validation_target, validation_group = v2.predict_residual(
                model,
                channel_builder,
                validation_loader,
                device,
            )
            if config.structure_on_final:
                validation_anchor = anchor_standardized[validation_indices]
                final_validation_prediction = validation_anchor + validation_prediction
                final_validation_target = validation_anchor + validation_target
            else:
                final_validation_prediction = validation_prediction
                final_validation_target = validation_target
            absolute_rmse = float(np.sqrt(np.mean((final_validation_prediction - final_validation_target) ** 2)))
            centered_values = []
            for value in np.unique(validation_group):
                mask = validation_group == value
                prediction_centered = final_validation_prediction[mask] - final_validation_prediction[mask].mean()
                target_centered = final_validation_target[mask] - final_validation_target[mask].mean()
                centered_values.append(float(np.sqrt(np.mean((prediction_centered - target_centered) ** 2))))
            score = absolute_rmse + 0.20 * float(np.mean(centered_values))
            row["validation_residual_score"] = score
            if epoch >= config.min_epochs and score < best_score - 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                without_improvement = 0
            elif epoch >= config.min_epochs:
                without_improvement += 1
            if epoch >= config.min_epochs and without_improvement >= config.patience:
                history.append(row)
                break
        history.append(row)

    if validation_loader is not None:
        if best_state is None:
            raise RuntimeError("No valid domain-robust checkpoint was recorded")
        model.load_state_dict(best_state)
        model.to(device)
    return model, channel_builder, history, int(best_epoch)


def argument_value(arguments: list[str], name: str) -> str:
    if name not in arguments:
        raise ValueError(f"{name} is required by the V8 wrapper")
    return arguments[arguments.index(name) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--domain-objective", choices=["group_dro", "vrex"], required=True)
    parser.add_argument("--robust-weight", type=float, default=0.35)
    parser.add_argument("--group-dro-temperature", type=float, default=0.20)
    parser.add_argument("--vrex-weight", type=float, default=0.25)
    variant, remaining = parser.parse_known_args()
    global DOMAIN_OBJECTIVE, ROBUST_WEIGHT, GROUP_DRO_TEMPERATURE, VREX_WEIGHT
    DOMAIN_OBJECTIVE = variant.domain_objective
    ROBUST_WEIGHT = variant.robust_weight
    GROUP_DRO_TEMPERATURE = variant.group_dro_temperature
    VREX_WEIGHT = variant.vrex_weight
    if not 0.0 <= ROBUST_WEIGHT <= 1.0:
        raise ValueError("--robust-weight must be in [0, 1]")
    if GROUP_DRO_TEMPERATURE <= 0.0 or VREX_WEIGHT < 0.0:
        raise ValueError("Domain robustness coefficients must be non-negative and temperature positive")

    original_argv = sys.argv
    original_trainer = v5.train_auxiliary_initialized_residual_model
    try:
        v5.train_auxiliary_initialized_residual_model = train_domain_robust_residual_model
        sys.argv = [sys.argv[0], *remaining]
        v5.main()
    finally:
        sys.argv = original_argv
        v5.train_auxiliary_initialized_residual_model = original_trainer

    output_dir = Path(argument_value(remaining, "--output-dir")).resolve()
    report = {
        "model": f"PLUMRAC-DG V8 {DOMAIN_OBJECTIVE}",
        "base_model": "PLUMRAC-MT V5.1",
        "domain_objective": DOMAIN_OBJECTIVE,
        "robust_weight": ROBUST_WEIGHT,
        "group_dro_temperature": GROUP_DRO_TEMPERATURE,
        "vrex_weight": VREX_WEIGHT,
        "domain_labels": "source-cultivar identity during training only",
        "heldout_cultivar_used_during_training": False,
        "final_model_outputs": 1,
        "final_model_is_trait_specific": True,
        "provenance_sha256": {
            "v8_domain_robust_trainer": sha256_file(Path(__file__).resolve()),
            "v5_auxiliary_trainer": sha256_file(Path(v5.__file__).resolve()),
            "v4_multiscale_architecture": sha256_file(Path(v4.__file__).resolve()),
        },
        "claim_boundary": (
            "Domain robust losses and all auxiliary pretraining use source training cultivars only in every "
            "nested LOCO stage. The deployed model has one output for one trait."
        ),
    }
    (output_dir / "v8_domain_robust.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model"] = report["model"]
    summary["v8_domain_robust"] = report
    summary["provenance_sha256"]["src\\train_plumrac_v8_domainrobust.py"] = report[
        "provenance_sha256"
    ]["v8_domain_robust_trainer"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
