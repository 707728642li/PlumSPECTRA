from __future__ import annotations

import argparse
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


MASK_RATE = 0.15
MASK_BLOCK_LENGTH = 9
RECONSTRUCTION_WEIGHT = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MaskedAuxiliaryPretrainNet(nn.Module):
    """Source-only auxiliary pretrainer; both heads are discarded after transfer."""

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
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width, bias=False),
            v4.group_norm(width),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, channels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.encode(channels)
        auxiliary = self.auxiliary_head(self.backbone.pool_features(features))
        reconstruction = self.reconstruction_head(features).squeeze(1)
        return auxiliary, reconstruction


def contiguous_mask(batch: int, length: int, device: torch.device) -> torch.Tensor:
    block = min(MASK_BLOCK_LENGTH, length)
    blocks = max(1, math.ceil(MASK_RATE * length / block))
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    offsets = torch.arange(block, device=device)[None, :]
    for _ in range(blocks):
        starts = torch.randint(0, length - block + 1, (batch, 1), device=device)
        mask.scatter_(1, starts + offsets, True)
    return mask


def mask_raw_spectrum(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A broad local baseline preserves physically plausible absorbance scale while
    # removing narrow-band information that the encoder must reconstruct.
    smooth = F.avg_pool1d(
        F.pad(raw[:, None, :], (10, 10), mode="reflect"),
        kernel_size=21,
        stride=1,
    ).squeeze(1)
    return torch.where(mask, smooth, raw)


def pretrain_encoder_masked(
    raw: np.ndarray,
    channel_builder: nn.Module,
    groups: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    config: v2.RACConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    if v5.GLOBAL_AUXILIARY_Y is None:
        raise RuntimeError("Auxiliary targets were not initialized")
    auxiliary = v5.GLOBAL_AUXILIARY_Y
    train_values = auxiliary[train_indices]
    target_mean = np.nanmean(train_values, axis=0, keepdims=True)
    target_sd = np.nanstd(train_values, axis=0, ddof=1, keepdims=True)
    target_sd = np.where(np.isfinite(target_sd) & (target_sd > 1e-6), target_sd, 1.0)
    standardized = (auxiliary - target_mean) / target_sd
    target_mask = np.isfinite(standardized)
    standardized = np.where(target_mask, standardized, 0.0).astype(np.float32)

    network = MaskedAuxiliaryPretrainNet(
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
    trainable = [
        *encoder_parameters,
        *network.auxiliary_head.parameters(),
        *network.reconstruction_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=v5.PRETRAIN_LEARNING_RATE,
        weight_decay=v5.PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(v5.PRETRAIN_EPOCHS, 1),
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sampler = v2.cultivar_balanced_sampler(groups, train_indices, config.sampler_power, seed + 701)
    loader = DataLoader(
        v5.AuxiliaryDataset(raw, standardized, target_mask, train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    history: list[float] = []
    for _ in range(v5.PRETRAIN_EPOCHS):
        network.train()
        epoch_losses: list[float] = []
        for raw_batch, target_batch, observed_batch in loader:
            raw_batch = raw_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            observed_batch = observed_batch.to(device, non_blocking=True)
            spectral_mask = contiguous_mask(raw_batch.shape[0], raw_batch.shape[1], device)
            masked_raw = mask_raw_spectrum(raw_batch, spectral_mask)
            with torch.no_grad():
                clean_raw_channel = channel_builder(raw_batch, augment=False)[:, 0]
            channels = channel_builder(masked_raw, augment=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction, reconstruction = network(channels)
                element_loss = F.smooth_l1_loss(prediction, target_batch, beta=0.5, reduction="none")
                target_losses = []
                for target_index in range(element_loss.shape[1]):
                    observed = observed_batch[:, target_index]
                    if observed.any():
                        target_losses.append(element_loss[observed, target_index].mean())
                auxiliary_loss = torch.stack(target_losses).mean()
                reconstruction_loss = F.smooth_l1_loss(
                    reconstruction[spectral_mask],
                    clean_raw_channel[spectral_mask],
                    beta=0.5,
                )
                loss = auxiliary_loss + RECONSTRUCTION_WEIGHT * reconstruction_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
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


def argument_value(arguments: list[str], name: str) -> str:
    if name not in arguments:
        raise ValueError(f"{name} is required by the V7 wrapper")
    return arguments[arguments.index(name) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--mask-block-length", type=int, default=9)
    parser.add_argument("--reconstruction-weight", type=float, default=0.15)
    variant, remaining = parser.parse_known_args()
    global MASK_RATE, MASK_BLOCK_LENGTH, RECONSTRUCTION_WEIGHT
    MASK_RATE = variant.mask_rate
    MASK_BLOCK_LENGTH = variant.mask_block_length
    RECONSTRUCTION_WEIGHT = variant.reconstruction_weight
    if not 0.0 < MASK_RATE < 1.0:
        raise ValueError("--mask-rate must be between zero and one")
    if MASK_BLOCK_LENGTH < 1:
        raise ValueError("--mask-block-length must be positive")
    if RECONSTRUCTION_WEIGHT <= 0.0:
        raise ValueError("--reconstruction-weight must be positive")

    original_argv = sys.argv
    original_pretrainer = v5.pretrain_encoder
    try:
        v5.pretrain_encoder = pretrain_encoder_masked
        sys.argv = [sys.argv[0], *remaining]
        v5.main()
    finally:
        sys.argv = original_argv
        v5.pretrain_encoder = original_pretrainer

    output_dir = Path(argument_value(remaining, "--output-dir")).resolve()
    report = {
        "model": "PLUMRAC-MSM V7",
        "base_model": "PLUMRAC-MT V5.1",
        "mask_rate": MASK_RATE,
        "mask_block_length": MASK_BLOCK_LENGTH,
        "reconstruction_weight": RECONSTRUCTION_WEIGHT,
        "masked_reconstruction_is_source_only": True,
        "pretraining_heads_discarded": ["auxiliary_head", "reconstruction_head"],
        "final_model_outputs": 1,
        "final_model_is_trait_specific": True,
        "provenance_sha256": {
            "v7_masked_pretrainer": sha256_file(Path(__file__).resolve()),
            "v5_auxiliary_trainer": sha256_file(Path(v5.__file__).resolve()),
            "v4_multiscale_architecture": sha256_file(Path(v4.__file__).resolve()),
        },
        "claim_boundary": (
            "Masking, reconstruction, auxiliary labels, and preprocessing statistics use source training "
            "cultivars only in every nested LOCO stage. The deployed model has one output for one trait."
        ),
    }
    (output_dir / "v7_masked_pretraining.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model"] = "PLUMRAC-MSM V7"
    summary["v7_masked_pretraining"] = report
    summary["provenance_sha256"]["src\\train_plumrac_v7_maskedpretrain.py"] = report[
        "provenance_sha256"
    ]["v7_masked_pretrainer"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
