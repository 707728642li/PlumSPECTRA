from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_texture_pls_loco import DEFAULT_TARGETS, preprocess_all, select_configuration
from v2_registry import add_cultivar_code, abbreviated_trait


@dataclass(frozen=True)
class RACConfig:
    width: int = 32
    blocks: int = 3
    dropout: float = 0.12
    batch_size: int = 128
    max_epochs: int = 120
    min_epochs: int = 8
    patience: int = 20
    learning_rate: float = 4e-4
    weight_decay: float = 2e-3
    center_weight: float = 0.20
    rank_weight: float = 0.08
    rank_temperature: float = 0.30
    sampler_power: float = 0.50
    augmentation: bool = True
    augmentation_scale_sd: float = 0.015
    augmentation_offset_sd: float = 0.004
    augmentation_slope_sd: float = 0.004
    augmentation_noise_sd: float = 0.0015
    validation_cultivars: int = 5
    num_workers: int = 0
    structure_on_final: bool = True
    attention_tail: bool = True
    min_gate_improvement: float = 0.01
    min_gate_win_fraction: float = 1.00
    max_gate_worst_degradation: float = 0.00
    max_residual_gate: float = 0.50


class SpectrumDataset(Dataset):
    def __init__(
        self,
        raw: np.ndarray,
        residual_target: np.ndarray,
        anchor_standardized: np.ndarray,
        group_index: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.raw = torch.from_numpy(np.asarray(raw[indices], dtype=np.float32))
        self.residual_target = torch.from_numpy(np.asarray(residual_target[indices], dtype=np.float32))
        self.anchor_standardized = torch.from_numpy(np.asarray(anchor_standardized[indices], dtype=np.float32))
        self.group_index = torch.from_numpy(np.asarray(group_index[indices], dtype=np.int64))

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.raw[index],
            self.residual_target[index],
            self.anchor_standardized[index],
            self.group_index[index],
        )


class SpectralChannelBuilder(nn.Module):
    """Apply raw-space augmentation before deriving fold-normalized channels."""

    def __init__(
        self,
        wavelength: np.ndarray,
        channel_mean: np.ndarray,
        channel_sd: np.ndarray,
        config: RACConfig,
    ) -> None:
        super().__init__()
        delta = float(np.median(np.diff(wavelength)))
        # torch.conv1d performs cross-correlation, so the Savitzky-Golay
        # dot-product coefficients are required here (not convolution order).
        coefficients = savgol_coeffs(11, 2, deriv=1, delta=delta, use="dot").astype(np.float32)
        self.register_buffer("derivative_kernel", torch.from_numpy(coefficients).view(1, 1, -1))
        self.register_buffer("channel_mean", torch.from_numpy(channel_mean.astype(np.float32)))
        self.register_buffer("channel_sd", torch.from_numpy(channel_sd.astype(np.float32)))
        self.register_buffer("slope_axis", torch.linspace(-1.0, 1.0, len(wavelength)).view(1, -1))
        self.config = config

    def physical_augmentation(self, raw: torch.Tensor) -> torch.Tensor:
        batch = raw.shape[0]
        scale = 1.0 + self.config.augmentation_scale_sd * torch.randn(batch, 1, device=raw.device)
        offset = self.config.augmentation_offset_sd * torch.randn(batch, 1, device=raw.device)
        slope = self.config.augmentation_slope_sd * torch.randn(batch, 1, device=raw.device)
        noise = self.config.augmentation_noise_sd * torch.randn_like(raw)
        return raw * scale + offset + slope * self.slope_axis + noise

    def forward(self, raw: torch.Tensor, augment: bool) -> torch.Tensor:
        if augment and self.config.augmentation:
            raw = self.physical_augmentation(raw)
        sample_sd = raw.std(dim=1, correction=1, keepdim=True).clamp_min(1e-8)
        snv = (raw - raw.mean(dim=1, keepdim=True)) / sample_sd
        padded = F.pad(raw.unsqueeze(1), (5, 5), mode="reflect")
        derivative = F.conv1d(padded, self.derivative_kernel).squeeze(1)
        channels = torch.stack([raw, snv, derivative], dim=1)
        return (channels - self.channel_mean) / self.channel_sd


class ResidualSpectralBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=7, padding=3 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.block(value))


class PlumRACNet(nn.Module):
    """Plum Residual-Anchored Cross-cultivar Network for one continuous trait."""

    def __init__(self, width: int, blocks: int, dropout: float, attention_tail: bool = True) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, width, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8][:blocks]
        self.blocks = nn.Sequential(*(ResidualSpectralBlock(width, dilation, dropout) for dilation in dilations))
        self.segment_pool = nn.AdaptiveAvgPool1d(8)
        self.attention_tail = attention_tail
        if attention_tail:
            attention_width = max(width // 4, 4)
            self.attention_pool = nn.Sequential(
                nn.Conv1d(width, attention_width, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(attention_width, 1, kernel_size=1),
            )
        representation = width * (11 if attention_tail else 10) + 1
        self.trait_tail = nn.Sequential(
            nn.Linear(representation, width * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, 1),
        )
        # Near-zero initialization preserves the PLSR anchor at startup while
        # still allowing gradients to reach the spectral encoder immediately.
        nn.init.normal_(self.trait_tail[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.trait_tail[-1].bias)

    def forward(self, channels: torch.Tensor, anchor_standardized: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(channels))
        pooled_parts = [features.mean(dim=-1), features.amax(dim=-1), self.segment_pool(features).flatten(1)]
        if self.attention_tail:
            weights = torch.softmax(self.attention_pool(features), dim=-1)
            pooled_parts.append(torch.sum(features * weights, dim=-1))
        pooled = torch.cat(pooled_parts, dim=1)
        representation = torch.cat([pooled, anchor_standardized[:, None]], dim=1)
        return self.trait_tail(representation).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_integer(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_clean_channels(raw: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    raw32 = np.asarray(raw, dtype=np.float32)
    sample_sd = raw32.std(axis=1, ddof=1, keepdims=True)
    snv = (raw32 - raw32.mean(axis=1, keepdims=True)) / np.where(sample_sd > 1e-8, sample_sd, 1.0)
    derivative = savgol_filter(
        raw32,
        window_length=11,
        polyorder=2,
        deriv=1,
        delta=float(np.median(np.diff(wavelength))),
        axis=1,
        # Mirror padding matches the differentiable torch implementation used
        # during raw-space augmentation and avoids train/inference edge drift.
        mode="mirror",
    ).astype(np.float32)
    return np.stack([raw32, snv.astype(np.float32), derivative], axis=1)


def fit_channel_scaler(channels: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = channels[indices].mean(axis=0, keepdims=True)
    sd = channels[indices].std(axis=0, ddof=1, keepdims=True)
    return mean.astype(np.float32), np.where(sd > 1e-5, sd, 1.0).astype(np.float32)


def choose_validation_groups(groups: np.ndarray, heldout: str, seed: int, count: int) -> list[str]:
    candidates = sorted(set(groups.tolist()) - {heldout})
    rng = np.random.default_rng(seed + stable_integer(heldout))
    return sorted(rng.choice(candidates, size=min(count, len(candidates) - 1), replace=False).tolist())


def cultivar_balanced_sampler(groups: np.ndarray, indices: np.ndarray, power: float, seed: int) -> WeightedRandomSampler:
    counts = pd.Series(groups[indices]).value_counts()
    weights = np.asarray([counts[group] ** (-power) for group in groups[indices]], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(indices), replacement=True, generator=generator)


def centred_batch_loss(prediction: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for value in torch.unique(group):
        mask = group == value
        if int(mask.sum()) < 2:
            continue
        pred_c = prediction[mask] - prediction[mask].mean()
        target_c = target[mask] - target[mask].mean()
        losses.append(F.smooth_l1_loss(pred_c, target_c, beta=0.5))
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def pairwise_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for value in torch.unique(group):
        members = torch.nonzero(group == value, as_tuple=False).flatten()
        if len(members) < 2:
            continue
        order = members[torch.randperm(len(members), device=members.device)]
        left = order[0::2]
        right = order[1::2]
        pair_count = min(len(left), len(right))
        if pair_count == 0:
            continue
        delta_target = target[left[:pair_count]] - target[right[:pair_count]]
        delta_prediction = prediction[left[:pair_count]] - prediction[right[:pair_count]]
        informative = delta_target.abs() > 0.05
        if informative.any():
            signed_margin = delta_target[informative].sign() * delta_prediction[informative]
            losses.append(F.softplus(-signed_margin / temperature).mean())
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def predict_residual(
    model: PlumRACNet,
    channel_builder: SpectralChannelBuilder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    with torch.no_grad():
        for raw, target, anchor, group in loader:
            raw = raw.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            channels = channel_builder(raw, augment=False)
            outputs.append(model(channels, anchor).cpu().numpy())
            targets.append(target.numpy())
            groups.append(group.numpy())
    return np.concatenate(outputs), np.concatenate(targets), np.concatenate(groups)


def macro_normalized_rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    target_sd: float,
) -> float:
    values = []
    for group in np.unique(groups):
        mask = groups == group
        values.append(float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)) / target_sd))
    return float(np.mean(values))


def select_residual_gate(
    anchor: np.ndarray,
    residual_prediction: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    target_sd: float,
    min_relative_improvement: float,
    min_win_fraction: float,
    max_worst_degradation: float,
    max_residual_gate: float,
) -> tuple[float, list[dict[str, float]]]:
    candidates = [gate for gate in (0.0, 0.25, 0.50) if gate <= max_residual_gate]
    if candidates == [0.0]:
        raise ValueError("max_residual_gate must admit at least one positive gate candidate")
    rows = []
    for gate in candidates:
        prediction = anchor + gate * residual_prediction
        absolute = macro_normalized_rmse(truth, prediction, groups, target_sd)
        centred_scores = []
        group_improvements = []
        for group in np.unique(groups):
            mask = groups == group
            true_c = truth[mask] - np.mean(truth[mask])
            pred_c = prediction[mask] - np.mean(prediction[mask])
            centred_scores.append(float(np.sqrt(np.mean((true_c - pred_c) ** 2)) / target_sd))
            anchor_rmse = float(np.sqrt(np.mean((truth[mask] - anchor[mask]) ** 2)))
            candidate_rmse = float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)))
            group_improvements.append((anchor_rmse - candidate_rmse) / max(anchor_rmse, 1e-12))
        centred = float(np.mean(centred_scores))
        rows.append(
            {
                "gate": gate,
                "absolute_score": absolute,
                "centred_score": centred,
                "selection_score": absolute + 0.20 * centred,
                "group_win_fraction": float(np.mean(np.asarray(group_improvements) > 0)),
                "median_group_improvement": float(np.median(group_improvements)),
                "worst_group_improvement": float(np.min(group_improvements)),
            }
        )
    rows.sort(key=lambda row: (row["selection_score"], row["gate"]))
    zero = next(row for row in rows if row["gate"] == 0.0)
    for row in rows:
        row["relative_improvement_vs_zero"] = float(
            (zero["selection_score"] - row["selection_score"]) / zero["selection_score"]
        )
    eligible_rows = [
        row
        for row in rows
        if row["gate"] > 0
        and row["relative_improvement_vs_zero"] >= min_relative_improvement
        and row["group_win_fraction"] >= min_win_fraction
        and row["worst_group_improvement"] >= -max_worst_degradation
    ]
    selected = eligible_rows[0] if eligible_rows else zero
    for row in rows:
        row["selected"] = float(row is selected)
    rows = [selected] + [row for row in rows if row is not selected]
    return float(selected["gate"]), rows


def objective_profiles(base: RACConfig, requested: list[str]) -> list[tuple[str, RACConfig]]:
    """Return predeclared objective variants for fold-internal model selection."""
    definitions = {
        "absolute": replace(base, center_weight=0.0, rank_weight=0.0),
        "balanced": base,
        "ranking": replace(base, center_weight=0.35, rank_weight=0.15),
    }
    unknown = sorted(set(requested) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown PlumRAC objective profiles: {unknown}")
    return [(name, definitions[name]) for name in requested]


def fit_anchor(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    fit_indices: np.ndarray,
    preprocessing: str,
    n_components: int,
) -> PLSRegression:
    model = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
    model.fit(arrays[preprocessing][fit_indices], y[fit_indices])
    return model


def train_residual_model(
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
    config: RACConfig,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[PlumRACNet, SpectralChannelBuilder, list[dict[str, float]], int]:
    set_seed(seed)
    channel_mean, channel_sd = fit_channel_scaler(clean_channels, train_indices)
    channel_builder = SpectralChannelBuilder(wavelength, channel_mean, channel_sd, config).to(device)
    model = PlumRACNet(config.width, config.blocks, config.dropout, config.attention_tail).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = fixed_epochs if fixed_epochs is not None else config.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sampler = cultivar_balanced_sampler(groups, train_indices, config.sampler_power, seed)
    train_loader = DataLoader(
        SpectrumDataset(raw, residual_target, anchor_standardized, group_index, train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = None
    if validation_indices is not None:
        validation_loader = DataLoader(
            SpectrumDataset(raw, residual_target, anchor_standardized, group_index, validation_indices),
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
                # The deployment quantity is the final anchored trait value,
                # not the PLSR residual.  Apply cultivar-centred and ranking
                # objectives to that final value so the RAC tail preserves and
                # improves fruit-level discrimination already present in PLSR.
                if config.structure_on_final:
                    final_prediction = anchor_batch + prediction
                    final_target = anchor_batch + residual_batch
                else:
                    final_prediction = prediction
                    final_target = residual_batch
                centred = centred_batch_loss(final_prediction, final_target, group_batch)
                ranking = pairwise_rank_loss(final_prediction, final_target, group_batch, config.rank_temperature)
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
        }
        if validation_loader is not None:
            validation_prediction, validation_target, validation_group = predict_residual(
                model, channel_builder, validation_loader, device
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
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                without_improvement = 0
            elif epoch >= config.min_epochs:
                without_improvement += 1
            if epoch >= config.min_epochs and without_improvement >= config.patience:
                history.append(row)
                break
        history.append(row)

    if validation_loader is not None:
        if best_state is None:
            raise RuntimeError("No valid PlumRAC-Net checkpoint was recorded")
        model.load_state_dict(best_state)
        model.to(device)
    return model, channel_builder, history, int(best_epoch)


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    variance_true, variance_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = variance_true + variance_pred + (mean_true - mean_pred) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan,
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan,
        "ccc": concordance_correlation(y_true, y_pred),
    }


def within_cultivar_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    centred_true = predictions["y_true"] - predictions.groupby("cultivar_ascii")["y_true"].transform("mean")
    centred_pred = predictions["y_pred"] - predictions.groupby("cultivar_ascii")["y_pred"].transform("mean")
    per_group = []
    for _, group in predictions.groupby("cultivar_ascii", observed=True):
        if len(group) < 3 or group["y_pred"].std() == 0:
            continue
        per_group.append(
            {
                "pearson_r": float(pearsonr(group["y_true"], group["y_pred"]).statistic),
                "spearman_rho": float(spearmanr(group["y_true"], group["y_pred"]).statistic),
            }
        )
    return {
        "centered_r2": float(r2_score(centred_true, centred_pred)),
        "macro_pearson_r": float(np.mean([row["pearson_r"] for row in per_group])),
        "macro_spearman_rho": float(np.mean([row["spearman_rho"] for row in per_group])),
    }


def load_pls_selection(
    pls_results: Path | None,
    target: str,
    heldout: str,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    if pls_results is not None:
        path = pls_results / "selected_hyperparameters.csv"
        table = pd.read_csv(path)
        row = table.loc[(table["target"] == target) & (table["heldout_cultivar"] == heldout)]
        if len(row) != 1:
            raise ValueError(f"Expected one PLS selection row for {target}/{heldout}, found {len(row)}")
        return {"preprocessing": str(row.iloc[0]["preprocessing"]), "n_components": int(row.iloc[0]["n_components"])}
    selected, _ = select_configuration(arrays, y, groups, train_indices, inner_splits=4)
    return selected


def run_fold_seed(
    target: str,
    heldout: str,
    seed: int,
    raw: np.ndarray,
    wavelength: np.ndarray,
    clean_channels: np.ndarray,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    eligible: np.ndarray,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    group_index: np.ndarray,
    pls_results: Path | None,
    output_dir: Path,
    config: RACConfig,
    profiles: list[str],
    device: torch.device,
) -> pd.DataFrame:
    fold_dir = output_dir / "runs" / abbreviated_trait(target) / heldout.replace(" ", "_") / f"seed_{seed}"
    prediction_path = fold_dir / "predictions.parquet"
    metadata_path = fold_dir / "metadata.json"
    if prediction_path.exists() and metadata_path.exists():
        return pd.read_parquet(prediction_path)
    fold_dir.mkdir(parents=True, exist_ok=True)

    outer_train = np.flatnonzero(eligible & (groups != heldout))
    test_indices = np.flatnonzero(eligible & (groups == heldout))
    validation_groups = choose_validation_groups(groups[eligible], heldout, seed, config.validation_cultivars)
    validation_indices = np.flatnonzero(eligible & np.isin(groups, validation_groups))
    inner_train = np.flatnonzero(eligible & (groups != heldout) & ~np.isin(groups, validation_groups))
    selection = load_pls_selection(pls_results, target, heldout, arrays, y, groups, outer_train)

    target_mean = float(np.mean(y[inner_train]))
    target_sd = max(float(np.std(y[inner_train], ddof=1)), 1e-6)
    anchor_inner = fit_anchor(arrays, y, inner_train, selection["preprocessing"], selection["n_components"])
    anchor_prediction = np.full(len(y), np.nan, dtype=np.float32)
    for indices in [inner_train, validation_indices]:
        anchor_prediction[indices] = anchor_inner.predict(arrays[selection["preprocessing"]][indices]).ravel()
    residual_target = np.zeros(len(y), dtype=np.float32)
    residual_target[inner_train] = ((y[inner_train] - anchor_prediction[inner_train]) / target_sd).astype(np.float32)
    residual_target[validation_indices] = ((y[validation_indices] - anchor_prediction[validation_indices]) / target_sd).astype(np.float32)
    anchor_standardized = np.zeros(len(y), dtype=np.float32)
    valid_anchor = np.isfinite(anchor_prediction)
    anchor_standardized[valid_anchor] = ((anchor_prediction[valid_anchor] - target_mean) / target_sd).astype(np.float32)

    validation_loader = DataLoader(
        SpectrumDataset(raw, residual_target, anchor_standardized, group_index, validation_indices),
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    profile_results: list[dict[str, Any]] = []
    for profile_name, profile_config in objective_profiles(config, profiles):
        profile_seed = seed + stable_integer(profile_name)
        model, channel_builder, history, best_epoch = train_residual_model(
            raw,
            wavelength,
            clean_channels,
            residual_target,
            anchor_standardized,
            group_index,
            groups,
            inner_train,
            validation_indices,
            profile_seed,
            profile_config,
            device,
        )
        validation_residual_z, _, _ = predict_residual(model, channel_builder, validation_loader, device)
        gate, gate_rows = select_residual_gate(
            anchor_prediction[validation_indices],
            validation_residual_z * target_sd,
            y[validation_indices],
            groups[validation_indices],
            target_sd,
            profile_config.min_gate_improvement,
            profile_config.min_gate_win_fraction,
            profile_config.max_gate_worst_degradation,
            profile_config.max_residual_gate,
        )
        pd.DataFrame(history).to_csv(fold_dir / f"selection_history_{profile_name}.csv", index=False)
        profile_results.append(
            {
                "profile": profile_name,
                "config": profile_config,
                "best_epoch": best_epoch,
                "gate": gate,
                "gate_rows": gate_rows,
                "selection_score": float(gate_rows[0]["selection_score"]),
            }
        )
    profile_results.sort(key=lambda item: (item["selection_score"], item["profile"]))
    selected_profile = profile_results[0]
    selected_config: RACConfig = selected_profile["config"]
    best_epoch = int(selected_profile["best_epoch"])
    gate = float(selected_profile["gate"])
    gate_rows = selected_profile["gate_rows"]

    final_mean = float(np.mean(y[outer_train]))
    final_sd = max(float(np.std(y[outer_train], ddof=1)), 1e-6)
    anchor_final = fit_anchor(arrays, y, outer_train, selection["preprocessing"], selection["n_components"])
    final_anchor_prediction = np.full(len(y), np.nan, dtype=np.float32)
    for indices in [outer_train, test_indices]:
        final_anchor_prediction[indices] = anchor_final.predict(arrays[selection["preprocessing"]][indices]).ravel()
    final_residual_target = np.zeros(len(y), dtype=np.float32)
    final_residual_target[outer_train] = ((y[outer_train] - final_anchor_prediction[outer_train]) / final_sd).astype(np.float32)
    final_anchor_standardized = np.zeros(len(y), dtype=np.float32)
    valid_final_anchor = np.isfinite(final_anchor_prediction)
    final_anchor_standardized[valid_final_anchor] = (
        (final_anchor_prediction[valid_final_anchor] - final_mean) / final_sd
    ).astype(np.float32)
    final_model, final_channel_builder, final_history, _ = train_residual_model(
        raw,
        wavelength,
        clean_channels,
        final_residual_target,
        final_anchor_standardized,
        group_index,
        groups,
        outer_train,
        None,
        seed + 1_000_003,
        selected_config,
        device,
        fixed_epochs=best_epoch,
    )
    test_loader = DataLoader(
        SpectrumDataset(raw, final_residual_target, final_anchor_standardized, group_index, test_indices),
        batch_size=selected_config.batch_size * 2,
        shuffle=False,
        num_workers=selected_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_residual_z, _, _ = predict_residual(final_model, final_channel_builder, test_loader, device)
    test_anchor = final_anchor_prediction[test_indices]
    prediction = test_anchor + gate * test_residual_z * final_sd

    frame = pd.DataFrame(
        {
            "sample_id": sample_ids[test_indices],
            "cultivar_ascii": groups[test_indices],
            "target": target,
            "seed": seed,
            "y_true": y[test_indices],
            "y_pred": prediction,
            "y_pls_anchor": test_anchor,
            "deep_residual": test_residual_z * final_sd,
            "residual_gate": gate,
        }
    )
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame = add_cultivar_code(frame)
    frame.to_parquet(prediction_path, index=False, compression="zstd")
    pd.DataFrame(final_history).to_csv(fold_dir / "retrain_history.csv", index=False)
    torch.save(final_model.state_dict(), fold_dir / "plumrac_state.pt")
    joblib.dump(anchor_final, fold_dir / "pls_anchor.joblib", compress=3)
    metadata = {
        "model": "PlumRAC-Net",
        "model_full_name": "Plum Residual-Anchored Cross-cultivar Network",
        "target": target,
        "trait_abbreviation": abbreviated_trait(target),
        "heldout_cultivar": heldout,
        "seed": seed,
        "validation_cultivars": validation_groups,
        "train_samples_selection": int(len(inner_train)),
        "validation_samples": int(len(validation_indices)),
        "train_samples_retrained": int(len(outer_train)),
        "test_samples": int(len(test_indices)),
        "selected_epoch": best_epoch,
        "selected_gate": gate,
        "selected_objective_profile": selected_profile["profile"],
        "gate_scores": gate_rows,
        "pls_anchor": selection,
        "config": asdict(selected_config),
        "objective_profile_candidates": [
            {
                "profile": item["profile"],
                "selection_score": item["selection_score"],
                "best_epoch": item["best_epoch"],
                "gate": item["gate"],
                "config": asdict(item["config"]),
            }
            for item in profile_results
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=DEFAULT_TARGETS)
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument(
        "--exclude-cultivars",
        default="",
        help="Comma-separated model-independent whole-cultivar QC exclusions.",
    )
    parser.add_argument("--heldout", default="all")
    parser.add_argument("--seeds", default="20260806")
    parser.add_argument("--pls-results", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--center-weight", type=float, default=0.20)
    parser.add_argument("--rank-weight", type=float, default=0.08)
    parser.add_argument("--sampler-power", type=float, default=0.50)
    parser.add_argument("--validation-cultivars", type=int, default=5)
    parser.add_argument("--min-gate-improvement", type=float, default=0.01)
    parser.add_argument("--min-gate-win-fraction", type=float, default=1.00)
    parser.add_argument("--max-gate-worst-degradation", type=float, default=0.00)
    parser.add_argument("--max-residual-gate", type=float, default=0.50)
    parser.add_argument(
        "--profiles",
        default="balanced",
        help="Comma-separated fold-internal objective profiles: absolute, balanced, ranking.",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--legacy-residual-structure-loss",
        action="store_true",
        help="Ablation: apply centred/rank objectives to the residual instead of the final anchored trait.",
    )
    parser.add_argument("--no-attention-tail", action="store_true", help="Ablation: disable attentive spectral pooling.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PlumRAC-Net V2 production training requires an available CUDA device")
    config = RACConfig(
        width=args.width,
        blocks=args.blocks,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        center_weight=args.center_weight,
        rank_weight=args.rank_weight,
        sampler_power=args.sampler_power,
        validation_cultivars=args.validation_cultivars,
        min_gate_improvement=args.min_gate_improvement,
        min_gate_win_fraction=args.min_gate_win_fraction,
        max_gate_worst_degradation=args.max_gate_worst_degradation,
        max_residual_gate=args.max_residual_gate,
        augmentation=not args.no_augmentation,
        structure_on_final=not args.legacy_residual_structure_loss,
        attention_tail=not args.no_attention_tail,
    )
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    y = pd.to_numeric(aligned[args.target], errors="coerce").to_numpy(float)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    group_names = sorted(np.unique(groups).tolist())
    excluded_cultivars = sorted(
        {value.strip() for value in args.exclude_cultivars.split(",") if value.strip()}
    )
    unknown_exclusions = sorted(set(excluded_cultivars) - set(group_names))
    if unknown_exclusions:
        raise ValueError(f"Unknown cultivar exclusions: {unknown_exclusions}")
    if excluded_cultivars:
        eligible &= ~np.isin(groups, excluded_cultivars)
    group_map = {group: index for index, group in enumerate(group_names)}
    group_index = np.asarray([group_map[group] for group in groups], dtype=np.int64)
    eligible_group_names = sorted(np.unique(groups[eligible]).tolist())
    heldout_groups = (
        eligible_group_names if args.heldout == "all" else [value.strip() for value in args.heldout.split(",")]
    )
    unavailable_heldout = sorted(set(heldout_groups) - set(eligible_group_names))
    if unavailable_heldout:
        raise ValueError(f"Held-out cultivars are excluded or have no eligible samples: {unavailable_heldout}")
    seeds = [int(value.strip()) for value in args.seeds.split(",")]
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    objective_profiles(config, profiles)
    clean_channels = build_clean_channels(raw, wavelength)
    arrays = preprocess_all(raw, wavelength)

    frames = []
    for heldout in heldout_groups:
        for seed in seeds:
            frame = run_fold_seed(
                args.target,
                heldout,
                seed,
                raw,
                wavelength,
                clean_channels,
                arrays,
                y,
                eligible,
                sample_ids,
                groups,
                group_index,
                args.pls_results.resolve() if args.pls_results else None,
                output_dir,
                config,
                profiles,
                device,
            )
            frames.append(frame)
            print(f"completed {abbreviated_trait(args.target)} / {heldout} / seed {seed}", flush=True)

    by_seed = pd.concat(frames, ignore_index=True)
    ensemble = (
        by_seed.groupby(["sample_id", "cultivar_ascii", "cultivar_code", "target"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            y_pls_anchor=("y_pls_anchor", "mean"),
            prediction_sd=("y_pred", "std"),
            seeds=("seed", "nunique"),
        )
    )
    ensemble["residual"] = ensemble["y_pred"] - ensemble["y_true"]
    by_seed.to_parquet(output_dir / "predictions_by_seed.parquet", index=False, compression="zstd")
    ensemble.to_parquet(output_dir / "predictions_ensemble.parquet", index=False, compression="zstd")
    fold_rows = []
    for (cultivar, seed), group in by_seed.groupby(["cultivar_ascii", "seed"], observed=True):
        fold_rows.append(
            {
                "target": args.target,
                "trait_abbreviation": abbreviated_trait(args.target),
                "heldout_cultivar": cultivar,
                "seed": seed,
                **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics_by_seed.csv", index=False)
    summary = {
        "model": "PlumRAC-Net",
        "model_full_name": "Plum Residual-Anchored Cross-cultivar Network",
        "validation": "nested leave-one-cultivar-out with final full-outer-training refit",
        "target": args.target,
        "trait_abbreviation": abbreviated_trait(args.target),
        "cohort": args.cohort,
        "model_independent_excluded_cultivars": excluded_cultivars,
        "eligible_cultivars": int(len(eligible_group_names)),
        "eligible_samples": int(eligible.sum()),
        "qc_versions": sorted(aligned["qc_version"].dropna().astype(str).unique().tolist()),
        "seeds": seeds,
        "config": asdict(config),
        "pooled_metrics": regression_metrics(ensemble["y_true"].to_numpy(), ensemble["y_pred"].to_numpy()),
        "pls_anchor_pooled_metrics": regression_metrics(
            ensemble["y_true"].to_numpy(), ensemble["y_pls_anchor"].to_numpy()
        ),
        "within_cultivar_metrics": within_cultivar_metrics(ensemble),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in PlumRACNet(
                    config.width, config.blocks, config.dropout, config.attention_tail
                ).parameters()
            )
        ),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    project_root = Path(__file__).resolve().parents[1]
    provenance_paths = [
        Path(__file__).resolve(),
        args.qc_ledger.resolve(),
        multimodal_dir / "nir_c_absorbance.npy",
        multimodal_dir / "wavelength_nm.npy",
        multimodal_dir / "nir_c_row_index.csv",
        project_root / "configs" / "v2_nomenclature.csv",
        project_root / "configs" / "v2_trait_registry.csv",
        project_root / "configs" / "v2_plumrac_production.json",
        project_root / "environment-lock.txt",
    ]
    if args.pls_results:
        provenance_paths.append(args.pls_results.resolve() / "selected_hyperparameters.csv")
    summary["provenance_sha256"] = {
        str(path.relative_to(project_root) if path.is_relative_to(project_root) else path): sha256_file(path)
        for path in provenance_paths
        if path.exists()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
