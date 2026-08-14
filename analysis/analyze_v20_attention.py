from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4


TRAITS = ["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ai_dir = args.ai_dir.resolve()
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    sample_ids = row_index["sample_id"].astype(str).to_numpy()
    manifest = pd.read_csv(args.fold_manifest.resolve(), dtype={"sample_id": str})
    fold_by_sample = manifest.set_index("sample_id")["outer_fold"]
    aligned_folds = fold_by_sample.reindex(sample_ids).fillna(-1).to_numpy(int)

    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    rows: list[dict[str, object]] = []
    for trait in TRAITS:
        for fold in range(1, 6):
            run_dir = ai_dir / trait / f"fold_{fold}"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            config = v2.RACConfig(**metadata["config"])
            builder_state = torch.load(
                run_dir / "channel_builder_state.pt", map_location="cpu", weights_only=True
            )
            builder = v2.SpectralChannelBuilder(
                wavelength,
                builder_state["channel_mean"].numpy(),
                builder_state["channel_sd"].numpy(),
                config,
            )
            builder.load_state_dict(builder_state)
            builder.eval()
            model = v4.V4PlumRACNet(
                config.width, config.blocks, config.dropout, config.attention_tail
            )
            model.load_state_dict(
                torch.load(run_dir / "plumrac_state.pt", map_location="cpu", weights_only=True)
            )
            model.eval()
            test_indices = np.flatnonzero(aligned_folds == fold)
            attention_parts = []
            view_parts = []
            with torch.no_grad():
                for start in range(0, len(test_indices), 512):
                    indices = test_indices[start : start + 512]
                    channels = builder(torch.from_numpy(raw[indices]), augment=False)
                    summary = torch.cat(
                        [channels.mean(dim=-1), channels.std(dim=-1, correction=0)], dim=1
                    )
                    view_weights = 2.0 * model.view_gate.network(summary)
                    features = model.encode(channels)
                    weights = torch.softmax(model.attention_pool(features), dim=-1).squeeze(1)
                    attention_parts.append(weights.numpy())
                    view_parts.append(view_weights.numpy())
            attention = np.concatenate(attention_parts)
            view_weights = np.concatenate(view_parts)
            enrichment = attention.mean(axis=0) * attention.shape[1]
            for wl, value in zip(wavelength, enrichment, strict=True):
                rows.append(
                    {
                        "trait": trait,
                        "outer_fold": fold,
                        "wavelength_nm": float(wl),
                        "attention_enrichment": float(value),
                        "raw_view_weight": float(view_weights[:, 0].mean()),
                        "snv_view_weight": float(view_weights[:, 1].mean()),
                        "sg1_view_weight": float(view_weights[:, 2].mean()),
                    }
                )
            print(f"completed attention {trait} fold {fold}", flush=True)

    fold_attention = pd.DataFrame(rows)
    fold_attention.to_parquet(
        output_dir / "attention_by_trait_fold.parquet", index=False, compression="zstd"
    )
    summary = (
        fold_attention.groupby(["trait", "wavelength_nm"], as_index=False, observed=True)
        .agg(
            attention_median=("attention_enrichment", "median"),
            attention_q25=("attention_enrichment", lambda values: values.quantile(0.25)),
            attention_q75=("attention_enrichment", lambda values: values.quantile(0.75)),
            fold_cv=("attention_enrichment", lambda values: float(values.std(ddof=1) / max(values.mean(), 1e-12))),
        )
    )
    summary.to_csv(output_dir / "attention_wavelength_summary.csv", index=False)
    view_summary = (
        fold_attention.groupby(["trait", "outer_fold"], observed=True)[
            ["raw_view_weight", "snv_view_weight", "sg1_view_weight"]
        ]
        .first()
        .reset_index()
    )
    view_summary.to_csv(output_dir / "spectral_view_weights.csv", index=False)

    edges = np.arange(900.0, 1750.1, 50.0)
    window_rows = []
    for trait, frame in summary.groupby("trait", observed=True):
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            selected = frame[
                (frame["wavelength_nm"] >= low) & (frame["wavelength_nm"] < high)
            ]
            if selected.empty:
                continue
            window_rows.append(
                {
                    "trait": trait,
                    "window_low_nm": low,
                    "window_high_nm": high,
                    "attention_median": float(selected["attention_median"].mean()),
                    "fold_cv_median": float(selected["fold_cv"].median()),
                }
            )
    windows = pd.DataFrame(window_rows)
    windows["rank"] = windows.groupby("trait", observed=True)["attention_median"].rank(
        ascending=False, method="min"
    )
    windows.to_csv(output_dir / "attention_windows.csv", index=False)
    windows[windows["rank"] <= 5].sort_values(["trait", "rank"]).to_csv(
        output_dir / "top_attention_windows.csv", index=False
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.8), sharex=True, constrained_layout=True)
    for ax, trait in zip(axes.ravel(), TRAITS, strict=True):
        frame = summary[summary["trait"] == trait].sort_values("wavelength_nm")
        x = frame["wavelength_nm"].to_numpy(float)
        ax.fill_between(
            x,
            frame["attention_q25"].to_numpy(float),
            frame["attention_q75"].to_numpy(float),
            color="#80A8C2",
            alpha=0.30,
            linewidth=0,
        )
        ax.plot(x, frame["attention_median"], color="#2878B5", lw=1.2)
        ax.axhline(1.0, color="#68737D", linestyle="--", lw=0.7)
        ax.axvspan(1685, wavelength.max(), color="#C8CDD0", alpha=0.25)
        ax.set_title(trait, loc="left")
        ax.set_ylabel("Attention / uniform")
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Fold-stable spectral attention across nine trait-specific deep models",
        fontsize=13.5,
        weight="bold",
    )
    for suffix in ["pdf", "png"]:
        fig.savefig(
            output_dir / f"figS_v20_attention_stability.{suffix}",
            dpi=420,
            bbox_inches="tight",
        )
    plt.close(fig)

    heat = summary.pivot(index="trait", columns="wavelength_nm", values="attention_median").reindex(TRAITS)
    fig, ax = plt.subplots(figsize=(11.2, 4.5), constrained_layout=True)
    image = ax.imshow(
        heat.to_numpy(),
        aspect="auto",
        extent=[wavelength.min(), wavelength.max(), len(TRAITS) - 0.5, -0.5],
        cmap="mako",
    )
    ax.set_yticks(np.arange(len(TRAITS)), TRAITS)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title("Cross-trait spectral attention atlas", loc="left")
    fig.colorbar(image, ax=ax, label="Median attention enrichment")
    for suffix in ["pdf", "png"]:
        fig.savefig(
            output_dir / f"figS_v20_attention_atlas.{suffix}",
            dpi=420,
            bbox_inches="tight",
        )
    plt.close(fig)

    audit = {
        "protocol": "V20 test-spectrum attention extraction; responses not used",
        "traits": 9,
        "outer_folds_per_trait": 5,
        "models": 45,
        "wavelengths": int(len(wavelength)),
        "attention_scale": "relative to uniform allocation; 1.0 is uniform",
        "terminal_edge_caution_nm": [1685.0, float(wavelength.max())],
        "interpretation_limit": "Attention is model-based association, not a causal or compound-specific assignment.",
    }
    (output_dir / "summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
