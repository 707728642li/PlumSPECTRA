from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_coeffs, savgol_filter
from torch import nn
import torch.nn.functional as F

import train_plumrac_loco as v2


CHANNEL_SET = "physics"
ARCHITECTURE = "multiscale"
MIXSTYLE_P = 0.50
MIXSTYLE_ALPHA = 0.30
CURVATURE_SD = 0.003
LOW_FREQUENCY_SD = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_channels() -> int:
    return 7 if CHANNEL_SET == "physics" else 3


def group_norm(width: int) -> nn.GroupNorm:
    groups = 8 if width % 8 == 0 else 4 if width % 4 == 0 else 1
    return nn.GroupNorm(groups, width)


def _numpy_snv(value: np.ndarray) -> np.ndarray:
    sd = value.std(axis=1, ddof=1, keepdims=True)
    return ((value - value.mean(axis=1, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)


def _numpy_detrended(value: np.ndarray) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, value.shape[1], dtype=np.float32)[None, :]
    centred = value - value.mean(axis=1, keepdims=True)
    slope = np.sum(centred * axis, axis=1, keepdims=True) / np.sum(axis * axis)
    residual = centred - slope * axis
    return _numpy_snv(residual)


def build_physics_clean_channels(raw: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Seven sample-wise views; no held-out cultivar statistics are used."""
    raw32 = np.asarray(raw, dtype=np.float32)
    delta = float(np.median(np.diff(wavelength)))
    snv = _numpy_snv(raw32)
    detrended = _numpy_detrended(raw32)
    sg1_short = savgol_filter(raw32, 7, 3, deriv=1, delta=delta, axis=1, mode="mirror").astype(np.float32)
    sg1_long = savgol_filter(raw32, 21, 3, deriv=1, delta=delta, axis=1, mode="mirror").astype(np.float32)
    smooth = savgol_filter(raw32, 41, 3, deriv=0, axis=1, mode="mirror").astype(np.float32)
    highpass = _numpy_snv(raw32 - smooth)
    sg2 = savgol_filter(raw32, 15, 3, deriv=2, delta=delta, axis=1, mode="mirror").astype(np.float32)
    return np.stack([raw32, snv, detrended, sg1_short, sg1_long, highpass, sg2], axis=1)


class PhysicsSpectralChannelBuilder(nn.Module):
    """Differentiable physics views derived after raw-space nuisance augmentation."""

    def __init__(
        self,
        wavelength: np.ndarray,
        channel_mean: np.ndarray,
        channel_sd: np.ndarray,
        config: v2.RACConfig,
    ) -> None:
        super().__init__()
        delta = float(np.median(np.diff(wavelength)))
        for name, window, derivative in [
            ("sg1_short", 7, 1),
            ("sg1_long", 21, 1),
            ("smooth", 41, 0),
            ("sg2", 15, 2),
        ]:
            coefficients = savgol_coeffs(
                window,
                3,
                deriv=derivative,
                delta=delta,
                use="dot",
            ).astype(np.float32)
            self.register_buffer(name, torch.from_numpy(coefficients).view(1, 1, -1))
        length = len(wavelength)
        axis = torch.linspace(-1.0, 1.0, length).view(1, -1)
        self.register_buffer("slope_axis", axis)
        self.register_buffer("quadratic_axis", axis.square() - axis.square().mean())
        self.register_buffer("smooth_axis", torch.sin(torch.pi * axis))
        self.register_buffer("channel_mean", torch.from_numpy(channel_mean.astype(np.float32)))
        self.register_buffer("channel_sd", torch.from_numpy(channel_sd.astype(np.float32)))
        self.config = config

    @staticmethod
    def apply_kernel(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padding = (kernel.shape[-1] - 1) // 2
        padded = F.pad(value.unsqueeze(1), (padding, padding), mode="reflect")
        return F.conv1d(padded, kernel).squeeze(1)

    @staticmethod
    def snv(value: torch.Tensor) -> torch.Tensor:
        sd = value.std(dim=1, correction=1, keepdim=True).clamp_min(1e-8)
        return (value - value.mean(dim=1, keepdim=True)) / sd

    def detrended(self, value: torch.Tensor) -> torch.Tensor:
        centred = value - value.mean(dim=1, keepdim=True)
        slope = torch.sum(centred * self.slope_axis, dim=1, keepdim=True) / torch.sum(
            self.slope_axis.square()
        )
        return self.snv(centred - slope * self.slope_axis)

    def physical_augmentation(self, raw: torch.Tensor) -> torch.Tensor:
        batch = raw.shape[0]
        scale = 1.0 + self.config.augmentation_scale_sd * torch.randn(batch, 1, device=raw.device)
        offset = self.config.augmentation_offset_sd * torch.randn(batch, 1, device=raw.device)
        slope = self.config.augmentation_slope_sd * torch.randn(batch, 1, device=raw.device)
        curvature = CURVATURE_SD * torch.randn(batch, 1, device=raw.device)
        low_frequency = LOW_FREQUENCY_SD * torch.randn(batch, 1, device=raw.device)
        noise = self.config.augmentation_noise_sd * torch.randn_like(raw)
        return (
            raw * scale
            + offset
            + slope * self.slope_axis
            + curvature * self.quadratic_axis
            + low_frequency * self.smooth_axis
            + noise
        )

    def forward(self, raw: torch.Tensor, augment: bool) -> torch.Tensor:
        if augment and self.config.augmentation:
            raw = self.physical_augmentation(raw)
        snv = self.snv(raw)
        smooth = self.apply_kernel(raw, self.smooth)
        channels = torch.stack(
            [
                raw,
                snv,
                self.detrended(raw),
                self.apply_kernel(raw, self.sg1_short),
                self.apply_kernel(raw, self.sg1_long),
                self.snv(raw - smooth),
                self.apply_kernel(raw, self.sg2),
            ],
            dim=1,
        )
        return (channels - self.channel_mean) / self.channel_sd


class PhysicsViewGate(nn.Module):
    """Sample-adaptive, interpretable weighting of the seven physics views."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels * 2, 8)
        self.network = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        summary = torch.cat([value.mean(dim=-1), value.std(dim=-1, correction=0)], dim=1)
        weights = 2.0 * self.network(summary)
        return value * weights.unsqueeze(-1)


class SpectralMixStyle(nn.Module):
    """Mix feature statistics to reduce cultivar/operator/batch style reliance."""

    def __init__(self, probability: float, alpha: float) -> None:
        super().__init__()
        self.probability = probability
        self.alpha = alpha

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0 or value.shape[0] < 2:
            return value
        if float(torch.rand((), device=value.device)) >= self.probability:
            return value
        mean = value.mean(dim=-1, keepdim=True)
        sd = value.var(dim=-1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        normalized = (value - mean) / sd
        beta = torch.distributions.Beta(
            torch.tensor(self.alpha, device=value.device),
            torch.tensor(self.alpha, device=value.device),
        )
        mixture = beta.sample((value.shape[0], 1, 1)).to(dtype=value.dtype)
        permutation = torch.randperm(value.shape[0], device=value.device)
        mixed_mean = mixture * mean + (1.0 - mixture) * mean[permutation]
        mixed_sd = mixture * sd + (1.0 - mixture) * sd[permutation]
        return normalized * mixed_sd + mixed_mean


class DepthwiseSpectralBranch(nn.Module):
    def __init__(self, width: int, kernel: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.branch = nn.Sequential(
            nn.Conv1d(
                width,
                width,
                kernel_size=kernel,
                padding=padding,
                dilation=dilation,
                groups=width,
                bias=False,
            ),
            nn.Conv1d(width, width, kernel_size=1, bias=False),
            group_norm(width),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.branch(value)


class MultiScaleSpectralBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = group_norm(width)
        self.branches = nn.ModuleList(
            DepthwiseSpectralBranch(width, kernel, dilation) for kernel in (3, 9, 21)
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(width * 3, width, kernel_size=1, bias=False),
            group_norm(width),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(torch.full((1, width, 1), 1e-2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.pre_norm(value)
        hidden = self.fuse(torch.cat([branch(normalized) for branch in self.branches], dim=1))
        return F.gelu(value + self.layer_scale * hidden)


class DilatedSpectralBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation, bias=False),
            group_norm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, bias=False),
            group_norm(width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.block(value))


class V4PlumRACNet(nn.Module):
    """Single-trait, physics-informed residual CNN with a fold-safe PLSR anchor."""

    def __init__(self, width: int, blocks: int, dropout: float, attention_tail: bool = True) -> None:
        super().__init__()
        self.view_gate = PhysicsViewGate(input_channels())
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels(), width, 9, padding=4, bias=False),
            group_norm(width),
            nn.GELU(),
        )
        block_class = MultiScaleSpectralBlock if ARCHITECTURE == "multiscale" else DilatedSpectralBlock
        dilations = [1, 2, 4, 8][:blocks]
        self.blocks = nn.ModuleList(block_class(width, dilation, dropout) for dilation in dilations)
        self.mixstyle = SpectralMixStyle(MIXSTYLE_P, MIXSTYLE_ALPHA)
        self.segment_pool = nn.AdaptiveAvgPool1d(8)
        self.attention_tail = attention_tail
        if attention_tail:
            attention_width = max(width // 4, 4)
            self.attention_pool = nn.Sequential(
                nn.Conv1d(width, attention_width, 1),
                nn.GELU(),
                nn.Conv1d(attention_width, 1, 1),
            )
        representation = width * (11 if attention_tail else 10) + 1
        self.trait_tail = nn.Sequential(
            nn.Linear(representation, width * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, 1),
        )
        nn.init.normal_(self.trait_tail[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.trait_tail[-1].bias)

    def encode(self, channels: torch.Tensor) -> torch.Tensor:
        features = self.mixstyle(self.stem(self.view_gate(channels)))
        for index, block in enumerate(self.blocks):
            features = block(features)
            if index == 1:
                features = self.mixstyle(features)
        return features

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        pooled_parts = [features.mean(dim=-1), features.amax(dim=-1), self.segment_pool(features).flatten(1)]
        if self.attention_tail:
            weights = torch.softmax(self.attention_pool(features), dim=-1)
            pooled_parts.append(torch.sum(features * weights, dim=-1))
        return torch.cat(pooled_parts, dim=1)

    def forward(self, channels: torch.Tensor, anchor_standardized: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_features(self.encode(channels))
        representation = torch.cat([pooled, anchor_standardized[:, None]], dim=1)
        return self.trait_tail(representation).squeeze(1)


def select_expanded_residual_gate(
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
    """Select 0--1 residual strength using source-only validation cultivars."""
    candidates = [gate for gate in (0.0, 0.25, 0.50, 0.75, 1.00) if gate <= max_residual_gate + 1e-12]
    if not any(gate > 0 for gate in candidates):
        raise ValueError("max_residual_gate must admit at least one positive gate candidate")
    rows: list[dict[str, float]] = []
    for gate in candidates:
        prediction = anchor + gate * residual_prediction
        absolute = v2.macro_normalized_rmse(truth, prediction, groups, target_sd)
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
    eligible = [
        row
        for row in rows
        if row["gate"] > 0
        and row["relative_improvement_vs_zero"] >= min_relative_improvement
        and row["group_win_fraction"] >= min_win_fraction
        and row["worst_group_improvement"] >= -max_worst_degradation
    ]
    selected = eligible[0] if eligible else zero
    for row in rows:
        row["selected"] = float(row is selected)
    return float(selected["gate"]), [selected] + [row for row in rows if row is not selected]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--channel-set", choices=["basic", "physics"], default="physics")
    parser.add_argument("--architecture", choices=["dilated", "multiscale"], default="multiscale")
    parser.add_argument("--mixstyle-p", type=float, default=0.50)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.30)
    parser.add_argument("--curvature-sd", type=float, default=0.003)
    parser.add_argument("--low-frequency-sd", type=float, default=0.002)
    variant, remaining = parser.parse_known_args()
    global CHANNEL_SET, ARCHITECTURE, MIXSTYLE_P, MIXSTYLE_ALPHA, CURVATURE_SD, LOW_FREQUENCY_SD
    CHANNEL_SET = variant.channel_set
    ARCHITECTURE = variant.architecture
    MIXSTYLE_P = variant.mixstyle_p
    MIXSTYLE_ALPHA = variant.mixstyle_alpha
    CURVATURE_SD = variant.curvature_sd
    LOW_FREQUENCY_SD = variant.low_frequency_sd
    if not 0.0 <= MIXSTYLE_P <= 1.0:
        raise ValueError("--mixstyle-p must be in [0, 1]")
    if MIXSTYLE_ALPHA <= 0:
        raise ValueError("--mixstyle-alpha must be positive")
    if "--output-dir" not in remaining:
        raise ValueError("--output-dir is required by the wrapped V2 trainer")
    output_dir = Path(remaining[remaining.index("--output-dir") + 1]).resolve()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        v2.PlumRACNet = V4PlumRACNet
        v2.select_residual_gate = select_expanded_residual_gate
        if CHANNEL_SET == "physics":
            v2.build_clean_channels = build_physics_clean_channels
            v2.SpectralChannelBuilder = PhysicsSpectralChannelBuilder
        v2.main()
    finally:
        sys.argv = original_argv

    width = int(remaining[remaining.index("--width") + 1]) if "--width" in remaining else 32
    blocks = int(remaining[remaining.index("--blocks") + 1]) if "--blocks" in remaining else 3
    dropout = float(remaining[remaining.index("--dropout") + 1]) if "--dropout" in remaining else 0.12
    probe = V4PlumRACNet(width, blocks, dropout)
    parameter_count = int(sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad))
    model_name = "PLUMRAC-MS V4" if CHANNEL_SET == "basic" and ARCHITECTURE == "multiscale" else "PLUMRAC-PHY V4"
    report = {
        "model": model_name,
        "development_cycle": "retrospective architecture and domain-generalization development",
        "single_trait_model": True,
        "channel_set": CHANNEL_SET,
        "input_channels": input_channels(),
        "architecture": ARCHITECTURE,
        "multiscale_kernels": [3, 9, 21] if ARCHITECTURE == "multiscale" else None,
        "mixstyle_probability": MIXSTYLE_P,
        "mixstyle_alpha": MIXSTYLE_ALPHA,
        "curvature_augmentation_sd": CURVATURE_SD,
        "low_frequency_augmentation_sd": LOW_FREQUENCY_SD,
        "residual_gate_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
        "trainable_parameters": parameter_count,
        "wrapped_trainer": str(Path(v2.__file__).resolve()),
        "provenance_sha256": {
            "v4_phy_trainer": sha256_file(Path(__file__).resolve()),
            "wrapped_v2_trainer": sha256_file(Path(v2.__file__).resolve()),
        },
        "claim_boundary": (
            "All cultivars have appeared in earlier development. V4 results are retrospective nested-LOCO evidence, "
            "not a newly untouched confirmation set; the held-out cultivar remains absent from each fold's training, "
            "preprocessing, hyperparameter selection, and gate selection."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v4_phy_variant.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["model"] = model_name
        summary["v4_variant"] = report
        summary["provenance_sha256"]["src\\train_plumrac_v4_phy.py"] = report["provenance_sha256"][
            "v4_phy_trainer"
        ]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
