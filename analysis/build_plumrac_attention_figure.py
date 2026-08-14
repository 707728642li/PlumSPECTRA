from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm

from train_plumrac_loco import PlumRACNet, RACConfig, SpectralChannelBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--pls-vip", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    target = str(summary["target"])
    abbreviation = str(summary["trait_abbreviation"])
    cohort = str(summary["cohort"])
    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & pd.to_numeric(aligned[target], errors="coerce").notna().to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()

    fold_rows = []
    for metadata_path in sorted((model_dir / "runs" / abbreviation).glob("*/seed_*/metadata.json")):
        run_dir = metadata_path.parent
        state_path = run_dir / "inference_state.npz"
        manifest_path = run_dir / "inference_manifest.json"
        if not state_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Run must be packaged before attention analysis: {run_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        heldout = str(metadata["heldout_cultivar"])
        indices = np.flatnonzero(eligible & (groups == heldout))
        state = np.load(state_path)
        config = RACConfig(**metadata["config"])
        model = PlumRACNet(config.width, config.blocks, config.dropout, config.attention_tail)
        model.load_state_dict(torch.load(run_dir / "plumrac_state.pt", map_location="cpu", weights_only=True))
        model.eval()
        builder = SpectralChannelBuilder(
            wavelength,
            state["channel_mean"],
            state["channel_sd"],
            config,
        )
        attention_parts = []
        with torch.no_grad():
            for start in range(0, len(indices), 512):
                batch_indices = indices[start : start + 512]
                channels = builder(torch.from_numpy(raw[batch_indices]), augment=False)
                features = model.blocks(model.stem(channels))
                weights = torch.softmax(model.attention_pool(features), dim=-1).squeeze(1)
                attention_parts.append(weights.numpy())
        attention = np.concatenate(attention_parts)
        # Express weight relative to uniform allocation: 1.0 is uniform.
        mean_enrichment = attention.mean(axis=0) * attention.shape[1]
        cultivar_code = pd.read_parquet(run_dir / "predictions.parquet")["cultivar_code"].iloc[0]
        fold_rows.extend(
            {
                "cultivar_code": cultivar_code,
                "heldout_cultivar": heldout,
                "seed": metadata["seed"],
                "wavelength_nm": float(wl),
                "attention_enrichment": float(value),
                "selected_gate": float(manifest["selected_gate"]),
            }
            for wl, value in zip(wavelength, mean_enrichment)
        )
    fold_attention = pd.DataFrame(fold_rows)
    fold_attention.to_parquet(output / "attention_by_fold.parquet", index=False, compression="zstd")
    attention_summary = (
        fold_attention.groupby("wavelength_nm", as_index=False)
        .agg(
            attention_median=("attention_enrichment", "median"),
            attention_q25=("attention_enrichment", lambda value: value.quantile(0.25)),
            attention_q75=("attention_enrichment", lambda value: value.quantile(0.75)),
            attention_mean=("attention_enrichment", "mean"),
        )
    )

    vip = None
    if args.pls_vip:
        vip_all = pd.read_csv(args.pls_vip.resolve())
        vip = vip_all.loc[vip_all["target"] == target, ["wavelength_nm", "vip_median"]].copy()
        attention_summary = attention_summary.merge(vip, on="wavelength_nm", how="left", validate="one_to_one")
    attention_summary.to_csv(output / "attention_wavelength_summary.csv", index=False)

    window_rows = []
    edges = np.arange(900, 1751, 50)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = attention_summary.loc[
            (attention_summary["wavelength_nm"] >= low) & (attention_summary["wavelength_nm"] < high)
        ]
        if selected.empty:
            continue
        row = {
            "window": f"{low}-{high}",
            "window_center_nm": (low + high) / 2,
            "attention": float(selected["attention_median"].mean()),
        }
        if vip is not None:
            row["pls_vip"] = float(selected["vip_median"].mean())
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "attention_windows.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    x = attention_summary["wavelength_nm"].to_numpy(float)
    axes[0, 0].fill_between(
        x,
        attention_summary["attention_q25"].to_numpy(float),
        attention_summary["attention_q75"].to_numpy(float),
        color="#80A8C2",
        alpha=0.30,
        linewidth=0,
    )
    axes[0, 0].plot(x, attention_summary["attention_median"], color="#2F6F98", linewidth=1.4)
    axes[0, 0].axhline(1.0, color="#6D767D", linestyle="--", linewidth=0.8)
    axes[0, 0].axvspan(1685, wavelength.max(), color="#B7BDC1", alpha=0.20)
    axes[0, 0].set_ylabel("Attention enrichment")
    axes[0, 0].set_title(f"A  {abbreviation} attentive RAC spectrum", loc="left", fontweight="bold")

    matrix = fold_attention.pivot_table(index="cultivar_code", columns="wavelength_nm", values="attention_enrichment", aggfunc="mean")
    image = axes[0, 1].imshow(
        matrix.to_numpy(),
        aspect="auto",
        extent=[wavelength.min(), wavelength.max(), len(matrix) - 0.5, -0.5],
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=1.0, vmin=float(matrix.to_numpy().min()), vmax=float(matrix.to_numpy().max())),
    )
    axes[0, 1].set_yticks(np.arange(len(matrix)), matrix.index)
    axes[0, 1].set_title("B  Held-out fold attention", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 1], label="Enrichment vs uniform")

    axes[1, 0].plot(x, attention_summary["attention_median"], color="#3F7CA6", label="PlumRAC attention")
    if vip is not None:
        axes[1, 0].plot(x, attention_summary["vip_median"], color="#C58C2A", label="PLSR VIP")
    axes[1, 0].axvspan(1685, wavelength.max(), color="#B7BDC1", alpha=0.20, label="terminal-edge caution")
    axes[1, 0].set_ylabel("Model importance (relative scale)")
    axes[1, 0].set_title("C  Nonlinear attention vs chemometric VIP", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=8)

    top = windows.nlargest(6, "attention").sort_values("attention")
    y = np.arange(len(top))
    axes[1, 1].barh(y, top["attention"], color="#3F7CA6")
    axes[1, 1].set_yticks(y, top["window"])
    axes[1, 1].set_xlabel("Mean attention enrichment")
    axes[1, 1].set_title("D  Top 50-nm attention windows", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.set_xlabel("Wavelength (nm)" if axis is not axes[1, 1] else axis.get_xlabel())
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.6)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_{abbreviation.lower()}_plumrac_attention.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "model": "PlumRAC-Net",
        "trait": abbreviation,
        "folds": int(fold_attention[["cultivar_code", "seed"]].drop_duplicates().shape[0]),
        "attention_scale": "relative to uniform allocation; 1.0 is uniform",
        "terminal_edge_caution_nm": [1685, float(wavelength.max())],
        "interpretation_limit": "Attention and VIP are model-based importance measures, not causal compound assignments.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
