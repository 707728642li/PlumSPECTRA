from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PRIMARY_TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]
AUXILIARY_TARGETS = [
    "skin_break_force_g_mean",
    "flesh_force_mean_g_mean",
    "loading_stiffness_g_per_rawpos_mean",
    "post_break_work_g_rawpos_mean",
]


@dataclass
class TrainingConfig:
    architecture: str
    primary_targets: list[str]
    auxiliary_texture: bool
    augmentation: bool
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    auxiliary_loss_weight: float
    validation_cultivars: int
    validation_mode: str
    validation_fraction: float
    loss: str


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> None:
        self.x = torch.from_numpy(x[indices])
        self.y = torch.from_numpy(y[indices])
        self.mask = torch.from_numpy(mask[indices])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.mask[index]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.10) -> None:
        super().__init__()
        padding = 3 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class MultiscaleCNN(nn.Module):
    def __init__(self, sequence_length: int, outputs: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(64, dilation) for dilation in [1, 2, 4, 8]))
        self.segment_pool = nn.AdaptiveAvgPool1d(16)
        self.head = nn.Sequential(
            nn.Linear(128 + 64 * 16, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, outputs),
        )
        self.linear_skip = nn.Linear(3 * sequence_length, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        pooled = torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1), self.segment_pool(features).flatten(1)], dim=1
        )
        return self.head(pooled) + self.linear_skip(x.flatten(1))


class PatchTransformer(nn.Module):
    def __init__(self, sequence_length: int, outputs: int) -> None:
        super().__init__()
        dimension = 96
        self.patch = nn.Conv1d(3, dimension, kernel_size=8, stride=4, bias=False)
        tokens = (sequence_length - 8) // 4 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.15), nn.Linear(128, outputs))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.position)
        return self.head(self.norm(encoded[:, 0]))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_integer(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def build_channels(absorbance: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    raw = np.asarray(absorbance, dtype=np.float32)
    sample_sd = raw.std(axis=1, ddof=1, keepdims=True)
    snv = (raw - raw.mean(axis=1, keepdims=True)) / np.where(sample_sd > 1e-8, sample_sd, 1.0)
    derivative = savgol_filter(
        raw,
        window_length=11,
        polyorder=2,
        deriv=1,
        delta=float(np.median(np.diff(wavelength))),
        axis=1,
        mode="interp",
    ).astype(np.float32)
    return np.stack([raw, snv.astype(np.float32), derivative], axis=1)


def fit_channel_normalization(channels: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = channels[train_indices].mean(axis=0, keepdims=True)
    sd = channels[train_indices].std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1e-5, sd, 1.0)
    return mean.astype(np.float32), sd.astype(np.float32)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    batch, _, length = x.shape
    scale = 1.0 + 0.025 * torch.randn(batch, 1, 1, device=x.device)
    offset = 0.020 * torch.randn(batch, 1, 1, device=x.device)
    slope_axis = torch.linspace(-1.0, 1.0, length, device=x.device).view(1, 1, length)
    slope = 0.015 * torch.randn(batch, 1, 1, device=x.device) * slope_axis
    noise = 0.010 * torch.randn_like(x)
    channel_dropout = (torch.rand(batch, 3, 1, device=x.device) > 0.03).to(x.dtype)
    return (x * scale + offset + slope + noise) * channel_dropout


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    variance_true, variance_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = variance_true + variance_pred + (mean_true - mean_pred) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    standard_deviation = float(np.std(y_true, ddof=1))
    iqr = float(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    correlation = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": correlation,
        "ccc": concordance_correlation(y_true, y_pred),
        "rpd": standard_deviation / rmse if rmse > 0 else np.nan,
        "rpiq": iqr / rmse if rmse > 0 else np.nan,
    }


@torch.no_grad()
def predict_standardized(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for x, _, _ in loader:
        predictions.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(predictions)


def validation_score(
    prediction_physical: np.ndarray,
    labels_physical: np.ndarray,
    label_mask: np.ndarray,
    groups: np.ndarray,
    target_sd: np.ndarray,
    primary_count: int,
) -> float:
    scores: list[float] = []
    for target_index in range(primary_count):
        for cultivar in np.unique(groups):
            valid = label_mask[:, target_index] & (groups == cultivar)
            if valid.sum() < 3:
                continue
            rmse = np.sqrt(np.mean((prediction_physical[valid, target_index] - labels_physical[valid, target_index]) ** 2))
            scores.append(float(rmse / target_sd[target_index]))
    return float(np.mean(scores))


def choose_validation_groups(all_groups: list[str], heldout: str, seed: int, count: int) -> list[str]:
    candidates = [group for group in all_groups if group != heldout]
    rng = np.random.default_rng(seed + stable_integer(heldout))
    return sorted(rng.choice(candidates, size=min(count, len(candidates) - 1), replace=False).tolist())


def train_one(
    config: TrainingConfig,
    heldout: str,
    seed: int,
    channels: np.ndarray,
    label_values: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    sample_ids: np.ndarray,
    groups: np.ndarray,
    all_groups: list[str],
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = output_dir / "runs" / heldout.replace(" ", "_") / f"seed_{seed}"
    prediction_path = run_dir / "predictions.parquet"
    history_path = run_dir / "history.csv"
    metadata_path = run_dir / "run_metadata.json"
    if prediction_path.exists() and history_path.exists() and metadata_path.exists():
        return pd.read_parquet(prediction_path), pd.read_csv(history_path), json.loads(metadata_path.read_text(encoding="utf-8"))

    set_seed(seed)
    primary_eligible = label_mask[:, : len(config.primary_targets)].any(axis=1)
    any_label_eligible = label_mask.any(axis=1)
    test_indices = np.flatnonzero((groups == heldout) & primary_eligible)
    outer_train_indices = np.flatnonzero((groups != heldout) & any_label_eligible)
    if config.validation_mode == "heldout_cultivars":
        validation_groups = choose_validation_groups(all_groups, heldout, seed, config.validation_cultivars)
        validation_indices = np.flatnonzero(np.isin(groups, validation_groups) & any_label_eligible)
        train_indices = np.flatnonzero((groups != heldout) & ~np.isin(groups, validation_groups) & any_label_eligible)
    elif config.validation_mode == "within_cultivar":
        train_indices, validation_indices = train_test_split(
            outer_train_indices,
            test_size=config.validation_fraction,
            random_state=seed + stable_integer(heldout),
            shuffle=True,
            stratify=groups[outer_train_indices],
        )
        validation_groups = sorted(np.unique(groups[validation_indices]).tolist())
    else:
        raise ValueError(f"Unknown validation mode: {config.validation_mode}")
    channel_mean, channel_sd = fit_channel_normalization(channels, train_indices)
    normalized = ((channels - channel_mean) / channel_sd).astype(np.float32)

    output_count = len(label_names)
    target_mean = np.zeros(output_count, dtype=np.float32)
    target_sd = np.ones(output_count, dtype=np.float32)
    standardized_labels = np.zeros_like(label_values, dtype=np.float32)
    for column in range(output_count):
        valid_train = label_mask[train_indices, column]
        values = label_values[train_indices, column][valid_train]
        target_mean[column] = np.mean(values)
        target_sd[column] = max(np.std(values, ddof=1), 1e-6)
        valid_all = label_mask[:, column]
        standardized_labels[valid_all, column] = (
            label_values[valid_all, column] - target_mean[column]
        ) / target_sd[column]

    train_group_counts = pd.Series(groups[train_indices]).value_counts()
    sample_weights = np.asarray([1.0 / train_group_counts[group] for group in groups[train_indices]], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_indices), replacement=True, generator=generator)
    train_loader = DataLoader(
        ArrayDataset(normalized, standardized_labels, label_mask.astype(np.bool_), train_indices),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ArrayDataset(normalized, standardized_labels, label_mask.astype(np.bool_), validation_indices),
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ArrayDataset(normalized, standardized_labels, label_mask.astype(np.bool_), test_indices),
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    if config.architecture == "cnn":
        model: nn.Module = MultiscaleCNN(normalized.shape[-1], output_count)
    elif config.architecture == "transformer":
        model = PatchTransformer(normalized.shape[-1], output_count)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_function: nn.Module
    if config.loss == "mse":
        loss_function = nn.MSELoss(reduction="none")
    elif config.loss == "huber":
        loss_function = nn.SmoothL1Loss(reduction="none", beta=0.5)
    else:
        raise ValueError(f"Unknown loss: {config.loss}")
    label_weights = torch.ones(output_count, dtype=torch.float32, device=device)
    if output_count > len(config.primary_targets):
        label_weights[len(config.primary_targets) :] = config.auxiliary_loss_weight

    best_score = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            if config.augmentation:
                x_batch = augment_batch(x_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(x_batch)
                element_loss = loss_function(prediction, y_batch)
                weights = mask_batch.to(element_loss.dtype) * label_weights.view(1, -1)
                loss = (element_loss * weights).sum() / weights.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        validation_standardized = predict_standardized(model, validation_loader, device)
        validation_physical = validation_standardized * target_sd[None, :] + target_mean[None, :]
        score = validation_score(
            validation_physical,
            label_values[validation_indices],
            label_mask[validation_indices],
            groups[validation_indices],
            target_sd,
            len(config.primary_targets),
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(batches, 1),
                "validation_macro_normalized_rmse": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("No best model state recorded")
    model.load_state_dict(best_state)
    model.to(device)
    test_standardized = predict_standardized(model, test_loader, device)
    test_physical = test_standardized * target_sd[None, :] + target_mean[None, :]
    prediction_rows: list[dict[str, Any]] = []
    for relative_index, absolute_index in enumerate(test_indices):
        for target_index, target in enumerate(config.primary_targets):
            if not label_mask[absolute_index, target_index]:
                continue
            prediction_rows.append(
                {
                    "sample_id": sample_ids[absolute_index],
                    "cultivar_ascii": heldout,
                    "target": target,
                    "seed": seed,
                    "y_true": float(label_values[absolute_index, target_index]),
                    "y_pred": float(test_physical[relative_index, target_index]),
                    "residual": float(test_physical[relative_index, target_index] - label_values[absolute_index, target_index]),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    history = pd.DataFrame(history_rows)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "architecture": config.architecture,
            "config": asdict(config),
            "heldout_cultivar": heldout,
            "validation_cultivars": validation_groups,
            "seed": seed,
            "label_names": label_names,
            "state_dict": best_state,
            "channel_mean": channel_mean,
            "channel_sd": channel_sd,
            "target_mean": target_mean,
            "target_sd": target_sd,
        },
        checkpoint_path,
    )
    metadata = {
        "architecture": config.architecture,
        "heldout_cultivar": heldout,
        "validation_cultivars": validation_groups,
        "seed": seed,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "best_epoch": best_epoch,
        "epochs_run": int(len(history)),
        "best_validation_macro_normalized_rmse": best_score,
        "checkpoint": checkpoint_path.as_posix(),
    }
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    history.to_csv(history_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return predictions, history, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["cnn", "transformer"], required=True)
    parser.add_argument("--primary-targets", default="all", help="Comma-separated targets or 'all'")
    parser.add_argument("--aux-texture", action="store_true")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--heldout", default="all", help="Comma-separated cultivar names or 'all'")
    parser.add_argument("--seeds", default="20260806")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.30)
    parser.add_argument("--validation-cultivars", type=int, default=3)
    parser.add_argument("--validation-mode", choices=["heldout_cultivars", "within_cultivar"], default="within_cultivar")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    args = parser.parse_args()

    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the deep LOCO experiment")

    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    master = pd.read_parquet(multimodal_dir / "master_samples.parquet").set_index("sample_id")
    aligned = master.loc[row_index["sample_id"]].reset_index()
    channels = build_channels(absorbance, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].to_numpy()
    all_groups = sorted(np.unique(groups).tolist())
    heldout_groups = all_groups if args.heldout == "all" else [value.strip() for value in args.heldout.split(",")]
    unknown = sorted(set(heldout_groups) - set(all_groups))
    if unknown:
        raise ValueError(f"Unknown held-out cultivars: {unknown}")
    seeds = [int(value.strip()) for value in args.seeds.split(",")]

    primary_targets = PRIMARY_TARGETS if args.primary_targets == "all" else [value.strip() for value in args.primary_targets.split(",")]
    invalid_targets = sorted(set(primary_targets) - set(PRIMARY_TARGETS))
    if invalid_targets:
        raise ValueError(f"Unknown primary targets: {invalid_targets}")
    label_names = primary_targets + (AUXILIARY_TARGETS if args.aux_texture else [])
    label_values = aligned[label_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    label_mask = np.isfinite(label_values)
    for index, target in enumerate(primary_targets):
        label_mask[:, index] &= aligned[f"{target}_valid"].to_numpy(bool)
    if args.aux_texture:
        auxiliary_valid = aligned["texture_dual_valid"].to_numpy(bool)
        label_mask[:, len(primary_targets) :] &= auxiliary_valid[:, None]

    config = TrainingConfig(
        architecture=args.architecture,
        primary_targets=primary_targets,
        auxiliary_texture=args.aux_texture,
        augmentation=args.augmentation,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        validation_cultivars=args.validation_cultivars,
        validation_mode=args.validation_mode,
        validation_fraction=args.validation_fraction,
        loss=args.loss,
    )
    all_predictions: list[pd.DataFrame] = []
    run_metadata: list[dict[str, Any]] = []
    for heldout in heldout_groups:
        for seed in seeds:
            predictions, _, metadata = train_one(
                config,
                heldout,
                seed,
                channels,
                label_values,
                label_mask,
                label_names,
                sample_ids,
                groups,
                all_groups,
                output_dir,
                device,
            )
            all_predictions.append(predictions)
            run_metadata.append(metadata)
            print(
                f"completed {args.architecture} held-out {heldout} seed {seed}: "
                f"best epoch {metadata['best_epoch']}, val {metadata['best_validation_macro_normalized_rmse']:.4f}"
            )

    predictions_seed = pd.concat(all_predictions, ignore_index=True)
    ensemble = (
        predictions_seed.groupby(["sample_id", "cultivar_ascii", "target"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), prediction_sd=("y_pred", "std"), seeds=("seed", "nunique"))
    )
    ensemble["residual"] = ensemble["y_pred"] - ensemble["y_true"]
    seed_metric_rows: list[dict[str, Any]] = []
    for (target, cultivar, seed), group in predictions_seed.groupby(["target", "cultivar_ascii", "seed"]):
        seed_metric_rows.append({"target": target, "heldout_cultivar": cultivar, "seed": seed, **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())})
    ensemble_metric_rows: list[dict[str, Any]] = []
    for (target, cultivar), group in ensemble.groupby(["target", "cultivar_ascii"]):
        ensemble_metric_rows.append({"target": target, "heldout_cultivar": cultivar, **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())})
    pooled_metrics = {
        target: regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for target, group in ensemble.groupby("target")
    }
    predictions_seed.to_parquet(output_dir / "predictions_by_seed.parquet", index=False, compression="zstd")
    ensemble.to_parquet(output_dir / "predictions_ensemble.parquet", index=False, compression="zstd")
    pd.DataFrame(seed_metric_rows).to_csv(output_dir / "fold_metrics_by_seed.csv", index=False)
    pd.DataFrame(ensemble_metric_rows).to_csv(output_dir / "fold_metrics_ensemble.csv", index=False)
    pd.DataFrame(run_metadata).to_csv(output_dir / "run_metadata.csv", index=False)
    summary = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "heldout_cultivars_completed": heldout_groups,
        "seeds": seeds,
        "pooled_ensemble_metrics": pooled_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
