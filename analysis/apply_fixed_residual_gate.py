from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(np.square(truth - prediction)))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
        "r2": float(r2_score(truth, prediction)),
        "pearson_r": float(np.corrcoef(truth, prediction)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.gate <= 1.0:
        raise ValueError("--gate must be in [0, 1]")

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    by_seed_path = run_dir / "predictions_by_seed.parquet"
    by_seed = pd.read_parquet(by_seed_path)
    required = {"deep_residual", "y_pls_anchor", "y_true", "sample_id", "cultivar_ascii", "seed"}
    missing = sorted(required - set(by_seed.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    by_seed["y_pred_original_fold_gate"] = by_seed["y_pred"]
    by_seed["residual_gate_original"] = by_seed["residual_gate"]
    by_seed["residual_gate"] = float(args.gate)
    by_seed["y_pred"] = by_seed["y_pls_anchor"] + float(args.gate) * by_seed["deep_residual"]
    by_seed["residual"] = by_seed["y_pred"] - by_seed["y_true"]
    seed_count = int(by_seed["seed"].nunique())

    keys = ["sample_id", "cultivar_ascii", "cultivar_code", "target"]
    ensemble = (
        by_seed.groupby(keys, as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            y_pls_anchor=("y_pls_anchor", "mean"),
            prediction_sd=("y_pred", "std"),
            seeds=("seed", "nunique"),
        )
    )
    ensemble["residual"] = ensemble["y_pred"] - ensemble["y_true"]
    truth = ensemble["y_true"].to_numpy(float)
    prediction = ensemble["y_pred"].to_numpy(float)
    anchor = ensemble["y_pls_anchor"].to_numpy(float)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_parquet(output_dir / "predictions_by_seed.parquet", index=False, compression="zstd")
    ensemble.to_parquet(output_dir / "predictions_ensemble.parquet", index=False, compression="zstd")
    source_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary = {
        **source_summary,
        "model": (
            "PLUMRAC-MT V5.1 fixed-gate ensemble"
            if seed_count > 1
            else "PLUMRAC-MT V5.1 fixed-gate prediction"
        ),
        "validation": (
            "source-validation-selected fixed residual gate applied to "
            f"{seed_count}-seed nested-LOCO prediction"
        ),
        "fixed_gate_seed_count": seed_count,
        "fixed_residual_gate": float(args.gate),
        "fixed_gate_selection_evidence": str(args.selection_evidence.resolve()),
        "pooled_metrics": metrics(truth, prediction),
        "pls_anchor_pooled_metrics": metrics(truth, anchor),
        "source_training_run": str(run_dir),
    }
    summary["pooled_rmse_improvement_pct"] = 100.0 * (
        summary["pls_anchor_pooled_metrics"]["rmse"] - summary["pooled_metrics"]["rmse"]
    ) / summary["pls_anchor_pooled_metrics"]["rmse"]
    summary["provenance_sha256"] = {
        **source_summary.get("provenance_sha256", {}),
        "source_predictions_by_seed": sha256_file(by_seed_path),
        "src\\apply_fixed_residual_gate.py": sha256_file(Path(__file__).resolve()),
        "fixed_gate_selection_evidence": sha256_file(args.selection_evidence.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "gate": float(args.gate),
        "selection_uses_target_test_predictions": False,
        "selection_evidence": str(args.selection_evidence.resolve()),
        "rationale": (
            "A single gate was selected by mean source-validation score in the prespecified V5 historical "
            "development suite. It replaces unstable per-seed/per-fold gate decisions."
        ),
        "claim_boundary": (
            "The fixed gate was chosen during retrospective development. Performance remains nested-LOCO, "
            "not untouched external validation."
        ),
    }
    (output_dir / "fixed_gate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
