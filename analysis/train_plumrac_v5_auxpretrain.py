from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4


TEXTURE_AUXILIARY_TARGETS = [
    "skin_break_force_g_mean",
    "skin_break_displacement_raw_mean",
    "skin_break_drop_g_mean",
    "flesh_force_mean_g_mean",
    "force_at_6_rawpos_g_mean",
    "loading_stiffness_g_per_rawpos_mean",
    "loading_work_g_rawpos_mean",
    "post_break_work_g_rawpos_mean",
    "adhesive_force_g_mean",
]
QUALITY_AUXILIARY_TARGETS = [*TEXTURE_AUXILIARY_TARGETS, "fruit_weight_g", "soluble_solids_pct", "ph"]

GLOBAL_AUXILIARY_Y: np.ndarray | None = None
GLOBAL_AUXILIARY_TARGETS: list[str] = []
PRETRAIN_EPOCHS = 8
PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AuxiliaryDataset(Dataset):
    def __init__(
        self,
        raw: np.ndarray,
        target: np.ndarray,
        target_mask: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.raw = torch.from_numpy(np.asarray(raw[indices], dtype=np.float32))
        self.target = torch.from_numpy(np.asarray(target[indices], dtype=np.float32))
        self.target_mask = torch.from_numpy(np.asarray(target_mask[indices], dtype=bool))

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.raw[index], self.target[index], self.target_mask[index]


class AuxiliaryPretrainNet(nn.Module):
    def __init__(self, width: int, blocks: int, dropout: float, targets: int, attention_tail: bool) -> None:
        super().__init__()
        self.backbone = v4.V4PlumRACNet(width, blocks, dropout, attention_tail)
        representation = width * (11 if attention_tail else 10)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(representation, width * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, targets),
        )

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encode(channels)
        return self.auxiliary_head(self.backbone.pool_features(features))


def pretrain_encoder(
    raw: np.ndarray,
    channel_builder: nn.Module,
    groups: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    config: v2.RACConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    if GLOBAL_AUXILIARY_Y is None:
        raise RuntimeError("Auxiliary targets were not initialized")
    auxiliary = GLOBAL_AUXILIARY_Y
    train_values = auxiliary[train_indices]
    target_mean = np.nanmean(train_values, axis=0, keepdims=True)
    target_sd = np.nanstd(train_values, axis=0, ddof=1, keepdims=True)
    target_sd = np.where(np.isfinite(target_sd) & (target_sd > 1e-6), target_sd, 1.0)
    standardized = (auxiliary - target_mean) / target_sd
    target_mask = np.isfinite(standardized)
    standardized = np.where(target_mask, standardized, 0.0).astype(np.float32)

    network = AuxiliaryPretrainNet(
        config.width,
        config.blocks,
        config.dropout,
        standardized.shape[1],
        config.attention_tail,
    ).to(device)
    encoder_parameters = [
        parameter
        for name, parameter in network.backbone.named_parameters()
        if not name.startswith("trait_tail")
    ]
    optimizer = torch.optim.AdamW(
        [*encoder_parameters, *network.auxiliary_head.parameters()],
        lr=PRETRAIN_LEARNING_RATE,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(PRETRAIN_EPOCHS, 1),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sampler = v2.cultivar_balanced_sampler(groups, train_indices, config.sampler_power, seed + 701)
    loader = DataLoader(
        AuxiliaryDataset(raw, standardized, target_mask, train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    history: list[float] = []
    for _ in range(PRETRAIN_EPOCHS):
        network.train()
        epoch_losses = []
        for raw_batch, target_batch, mask_batch in loader:
            raw_batch = raw_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            channels = channel_builder(raw_batch, augment=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = network(channels)
                element_loss = F.smooth_l1_loss(prediction, target_batch, beta=0.5, reduction="none")
                target_losses = []
                for target_index in range(element_loss.shape[1]):
                    observed = mask_batch[:, target_index]
                    if observed.any():
                        target_losses.append(element_loss[observed, target_index].mean())
                loss = torch.stack(target_losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *network.auxiliary_head.parameters()], 2.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history.append(float(np.mean(epoch_losses)))
    encoder_state = {
        key: value.detach().cpu()
        for key, value in network.backbone.state_dict().items()
        if not key.startswith("trait_tail")
    }
    return encoder_state, history


def train_auxiliary_initialized_residual_model(
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
    encoder_state, pretrain_history = pretrain_encoder(
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
        losses = []
        for raw_batch, residual_batch, anchor_batch, group_batch in train_loader:
            raw_batch = raw_batch.to(device, non_blocking=True)
            residual_batch = residual_batch.to(device, non_blocking=True)
            anchor_batch = anchor_batch.to(device, non_blocking=True)
            group_batch = group_batch.to(device, non_blocking=True)
            channels = channel_builder(raw_batch, augment=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(channels, anchor_batch)
                regression = F.smooth_l1_loss(prediction, residual_batch, beta=0.5)
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
        scheduler.step()
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
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
            centred_values = []
            for value in np.unique(validation_group):
                mask = validation_group == value
                pred_c = final_validation_prediction[mask] - final_validation_prediction[mask].mean()
                target_c = final_validation_target[mask] - final_validation_target[mask].mean()
                centred_values.append(float(np.sqrt(np.mean((pred_c - target_c) ** 2))))
            score = absolute_rmse + 0.20 * float(np.mean(centred_values))
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
            raise RuntimeError("No valid auxiliary-pretrained checkpoint was recorded")
        model.load_state_dict(best_state)
        model.to(device)
    return model, channel_builder, history, int(best_epoch)


def argument_value(arguments: list[str], name: str) -> str:
    if name not in arguments:
        raise ValueError(f"{name} is required by the V5 wrapper")
    return arguments[arguments.index(name) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--auxiliary-set", choices=["texture9", "quality12"], default="quality12")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-3)
    variant, remaining = parser.parse_known_args()
    global GLOBAL_AUXILIARY_Y, GLOBAL_AUXILIARY_TARGETS
    global PRETRAIN_EPOCHS, PRETRAIN_LEARNING_RATE, PRETRAIN_WEIGHT_DECAY
    PRETRAIN_EPOCHS = variant.pretrain_epochs
    PRETRAIN_LEARNING_RATE = variant.pretrain_learning_rate
    PRETRAIN_WEIGHT_DECAY = variant.pretrain_weight_decay
    if PRETRAIN_EPOCHS < 1:
        raise ValueError("--pretrain-epochs must be positive")
    GLOBAL_AUXILIARY_TARGETS = (
        TEXTURE_AUXILIARY_TARGETS if variant.auxiliary_set == "texture9" else QUALITY_AUXILIARY_TARGETS
    )

    multimodal_dir = Path(argument_value(remaining, "--multimodal-dir")).resolve()
    ledger_path = Path(argument_value(remaining, "--qc-ledger")).resolve()
    output_dir = Path(argument_value(remaining, "--output-dir")).resolve()
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(ledger_path).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    GLOBAL_AUXILIARY_Y = np.column_stack(
        [pd.to_numeric(aligned[target], errors="coerce").to_numpy(float) for target in GLOBAL_AUXILIARY_TARGETS]
    )

    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        v2.PlumRACNet = v4.V4PlumRACNet
        v2.select_residual_gate = v4.select_expanded_residual_gate
        v2.train_residual_model = train_auxiliary_initialized_residual_model
        v2.main()
    finally:
        sys.argv = original_argv

    report = {
        "model": "PLUMRAC-MT V5",
        "final_model_outputs": 1,
        "final_model_is_trait_specific": True,
        "auxiliary_pretraining_only_on_source_cultivars": True,
        "auxiliary_set": variant.auxiliary_set,
        "auxiliary_targets": GLOBAL_AUXILIARY_TARGETS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretrain_learning_rate": PRETRAIN_LEARNING_RATE,
        "pretrain_weight_decay": PRETRAIN_WEIGHT_DECAY,
        "encoder": "PLUMRAC-MS V4 basic three-view multiscale encoder",
        "provenance_sha256": {
            "v5_trainer": sha256_file(Path(__file__).resolve()),
            "v4_architecture": sha256_file(Path(v4.__file__).resolve()),
            "wrapped_v2_trainer": sha256_file(Path(v2.__file__).resolve()),
        },
        "claim_boundary": (
            "Every auxiliary label and normalization statistic is restricted to source training cultivars within "
            "each nested LOCO stage. The saved regression model has one output for one target trait."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v5_auxiliary_pretraining.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["model"] = "PLUMRAC-MT V5"
        summary["v5_auxiliary_pretraining"] = report
        summary["provenance_sha256"]["src\\train_plumrac_v5_auxpretrain.py"] = report[
            "provenance_sha256"
        ]["v5_trainer"]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
