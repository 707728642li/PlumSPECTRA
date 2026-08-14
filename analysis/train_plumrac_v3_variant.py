from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import savgol_coeffs, savgol_filter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_plumrac_loco as v2


ACTIVATION = "gelu"
NORMALIZATION = "group"
CHANNEL_SET = "basic"
ANCHOR_POLICY = "plsr"
ORIGINAL_FIT_ANCHOR = v2.fit_anchor


def input_channels() -> int:
    return 7 if CHANNEL_SET == "multiview" else 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inner_best_linear_selection(
    pls_results: Path | None,
    target: str,
    heldout: str,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, object]:
    if pls_results is None:
        raise ValueError("inner_best_linear requires --pls-results with the frozen nested PLSR table")
    pls_table = pd.read_csv(pls_results / "selected_hyperparameters.csv")
    pls_row = pls_table.loc[
        (pls_table["target"] == target) & (pls_table["heldout_cultivar"] == heldout)
    ]
    if len(pls_row) != 1:
        raise ValueError(f"Expected one PLSR row for {target}/{heldout}, found {len(pls_row)}")
    pls_row = pls_row.iloc[0]
    abbreviation = v2.abbreviated_trait(target)
    ridge_metadata = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "v2"
        / f"ridge_{abbreviation.lower()}"
        / "runs"
        / abbreviation
        / heldout.replace(" ", "_")
        / "metadata.json"
    )
    ridge = json.loads(ridge_metadata.read_text(encoding="utf-8"))["selected"]
    pls_score = float(pls_row["inner_macro_normalized_rmse"])
    ridge_score = float(ridge["macro_normalized_rmse"])
    common = {
        "anchor_policy": "inner_best_linear",
        "plsr_inner_macro_normalized_rmse": pls_score,
        "ridge_inner_macro_normalized_rmse": ridge_score,
    }
    if ridge_score < pls_score:
        return {
            **common,
            "anchor_model": "Ridge",
            "preprocessing": str(ridge["preprocessing"]),
            "n_components": f"ridge:{float(ridge['alpha']):g}",
            "alpha": float(ridge["alpha"]),
        }
    return {
        **common,
        "anchor_model": "PLSR",
        "preprocessing": str(pls_row["preprocessing"]),
        "n_components": int(pls_row["n_components"]),
    }


def fit_linear_anchor(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    fit_indices: np.ndarray,
    preprocessing: str,
    n_components: int | str,
):
    if isinstance(n_components, str) and n_components.startswith("ridge:"):
        alpha = float(n_components.split(":", maxsplit=1)[1])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(arrays[preprocessing][fit_indices], y[fit_indices])
        return model
    return ORIGINAL_FIT_ANCHOR(arrays, y, fit_indices, preprocessing, int(n_components))


def annotate_linear_anchor_outputs(output_dir: Path) -> None:
    if ANCHOR_POLICY != "inner_best_linear":
        return
    fold_models: dict[str, str] = {}
    for metadata_path in (output_dir / "runs").glob("*/*/seed_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fold_models[str(metadata["heldout_cultivar"])] = str(metadata["pls_anchor"].get("anchor_model", "PLSR"))
        prediction_path = metadata_path.with_name("predictions.parquet")
        prediction = pd.read_parquet(prediction_path)
        prediction["anchor_model"] = str(metadata["pls_anchor"].get("anchor_model", "PLSR"))
        prediction["y_linear_anchor"] = prediction["y_pls_anchor"]
        prediction.to_parquet(prediction_path, index=False, compression="zstd")
    for name in ["predictions_by_seed.parquet", "predictions_ensemble.parquet"]:
        path = output_dir / name
        frame = pd.read_parquet(path)
        frame["anchor_model"] = frame["cultivar_ascii"].map(fold_models)
        frame["y_linear_anchor"] = frame["y_pls_anchor"]
        frame.to_parquet(path, index=False, compression="zstd")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["linear_anchor_policy"] = ANCHOR_POLICY
    summary["anchor_models_by_heldout_cultivar"] = fold_models
    summary["linear_anchor_pooled_metrics"] = summary["pls_anchor_pooled_metrics"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def activation_module() -> nn.Module:
    if ACTIVATION == "gelu":
        return nn.GELU()
    if ACTIVATION == "silu":
        return nn.SiLU()
    if ACTIVATION == "relu":
        return nn.ReLU()
    raise ValueError(ACTIVATION)


class ChannelLayerNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


def normalization_module(width: int) -> nn.Module:
    if NORMALIZATION == "group":
        return nn.GroupNorm(8 if width >= 8 else 1, width)
    if NORMALIZATION == "layer":
        return ChannelLayerNorm(width)
    raise ValueError(NORMALIZATION)


class V3ResidualSpectralBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=7, padding=3 * dilation, dilation=dilation, bias=False)
        self.norm1 = normalization_module(width)
        self.activation1 = activation_module()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2 * dilation, dilation=dilation, bias=False)
        self.norm2 = normalization_module(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(value)
        hidden = self.norm1(hidden)
        hidden = self.activation1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        if ACTIVATION == "gelu":
            return F.gelu(value + hidden)
        if ACTIVATION == "silu":
            return F.silu(value + hidden)
        return F.relu(value + hidden)


class V3PlumRACNet(nn.Module):
    def __init__(self, width: int, blocks: int, dropout: float, attention_tail: bool = True) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels(), width, kernel_size=9, padding=4, bias=False),
            normalization_module(width),
            activation_module(),
        )
        dilations = [1, 2, 4, 8][:blocks]
        self.blocks = nn.Sequential(*(V3ResidualSpectralBlock(width, dilation, dropout) for dilation in dilations))
        self.segment_pool = nn.AdaptiveAvgPool1d(8)
        self.attention_tail = attention_tail
        if attention_tail:
            attention_width = max(width // 4, 4)
            self.attention_pool = nn.Sequential(
                nn.Conv1d(width, attention_width, kernel_size=1),
                activation_module(),
                nn.Conv1d(attention_width, 1, kernel_size=1),
            )
        representation = width * (11 if attention_tail else 10) + 1
        self.trait_tail = nn.Sequential(
            nn.Linear(representation, width * 3),
            activation_module(),
            nn.Dropout(dropout),
            nn.Linear(width * 3, 1),
        )
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


def build_multiview_clean_channels(raw: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    raw32 = np.asarray(raw, dtype=np.float32)
    delta = float(np.median(np.diff(wavelength)))
    sample_sd = raw32.std(axis=1, ddof=1, keepdims=True)
    snv = (raw32 - raw32.mean(axis=1, keepdims=True)) / np.where(sample_sd > 1e-8, sample_sd, 1.0)

    def derivative(value: np.ndarray, window: int, order: int) -> np.ndarray:
        return savgol_filter(
            value,
            window_length=window,
            polyorder=3 if window >= 7 else 2,
            deriv=order,
            delta=delta,
            axis=1,
            mode="mirror",
        ).astype(np.float32)

    return np.stack(
        [
            raw32,
            snv.astype(np.float32),
            derivative(raw32, 7, 1),
            derivative(raw32, 11, 1),
            derivative(raw32, 21, 1),
            derivative(raw32, 11, 2),
            derivative(snv, 11, 1),
        ],
        axis=1,
    )


class MultiviewSpectralChannelBuilder(nn.Module):
    """Differentiable, augmentation-consistent multi-scale chemometric views."""

    def __init__(
        self,
        wavelength: np.ndarray,
        channel_mean: np.ndarray,
        channel_sd: np.ndarray,
        config: v2.RACConfig,
    ) -> None:
        super().__init__()
        delta = float(np.median(np.diff(wavelength)))
        for name, window, order in [
            ("sg1_w7", 7, 1),
            ("sg1_w11", 11, 1),
            ("sg1_w21", 21, 1),
            ("sg2_w11", 11, 2),
        ]:
            coefficients = savgol_coeffs(window, 3, deriv=order, delta=delta, use="dot").astype(np.float32)
            self.register_buffer(name, torch.from_numpy(coefficients).view(1, 1, -1))
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

    @staticmethod
    def apply_kernel(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padding = (kernel.shape[-1] - 1) // 2
        return F.conv1d(F.pad(value.unsqueeze(1), (padding, padding), mode="reflect"), kernel).squeeze(1)

    def forward(self, raw: torch.Tensor, augment: bool) -> torch.Tensor:
        if augment and self.config.augmentation:
            raw = self.physical_augmentation(raw)
        sample_sd = raw.std(dim=1, correction=1, keepdim=True).clamp_min(1e-8)
        snv = (raw - raw.mean(dim=1, keepdim=True)) / sample_sd
        channels = torch.stack(
            [
                raw,
                snv,
                self.apply_kernel(raw, self.sg1_w7),
                self.apply_kernel(raw, self.sg1_w11),
                self.apply_kernel(raw, self.sg1_w21),
                self.apply_kernel(raw, self.sg2_w11),
                self.apply_kernel(snv, self.sg1_w11),
            ],
            dim=1,
        )
        return (channels - self.channel_mean) / self.channel_sd


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--activation", choices=["gelu", "silu", "relu"], default="gelu")
    parser.add_argument("--normalization", choices=["group", "layer"], default="group")
    parser.add_argument("--channel-set", choices=["basic", "multiview"], default="basic")
    parser.add_argument("--anchor-policy", choices=["plsr", "inner_best_linear"], default="plsr")
    variant, remaining = parser.parse_known_args()
    global ACTIVATION, NORMALIZATION, CHANNEL_SET, ANCHOR_POLICY
    ACTIVATION = variant.activation
    NORMALIZATION = variant.normalization
    CHANNEL_SET = variant.channel_set
    ANCHOR_POLICY = variant.anchor_policy
    if "--output-dir" not in remaining:
        raise ValueError("--output-dir is required by the wrapped V2 trainer")
    output_dir = Path(remaining[remaining.index("--output-dir") + 1]).resolve()
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        v2.PlumRACNet = V3PlumRACNet
        if CHANNEL_SET == "multiview":
            v2.build_clean_channels = build_multiview_clean_channels
            v2.SpectralChannelBuilder = MultiviewSpectralChannelBuilder
        if ANCHOR_POLICY == "inner_best_linear":
            v2.load_pls_selection = load_inner_best_linear_selection
            v2.fit_anchor = fit_linear_anchor
        v2.main()
    finally:
        sys.argv = original_argv
    output_dir.mkdir(parents=True, exist_ok=True)
    annotate_linear_anchor_outputs(output_dir)
    parameter_probe = V3PlumRACNet(
        width=int(remaining[remaining.index("--width") + 1]) if "--width" in remaining else 32,
        blocks=int(remaining[remaining.index("--blocks") + 1]) if "--blocks" in remaining else 3,
        dropout=float(remaining[remaining.index("--dropout") + 1]) if "--dropout" in remaining else 0.12,
    )
    report = {
        "development_cycle": "V3 architecture/operator ablation",
        "activation": ACTIVATION,
        "normalization": NORMALIZATION,
        "channel_set": CHANNEL_SET,
        "input_channels": input_channels(),
        "anchor_policy": ANCHOR_POLICY,
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameter_probe.parameters() if parameter.requires_grad)),
        "wrapped_trainer": str(Path(v2.__file__).resolve()),
        "provenance_sha256": {
            "v3_variant_trainer": sha256_file(Path(__file__).resolve()),
            "wrapped_v2_trainer": sha256_file(Path(v2.__file__).resolve()),
        },
        "claim_boundary": "Development-fold result only until the selected variant is frozen and evaluated on untouched confirmation folds.",
    }
    (output_dir / "v3_variant.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
