from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_texture_pls_loco import regression_metrics


REQUIRED_WIDE_EPSILON_POINTS = {1.5, 2.4, 4.0, 8.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def selected_search_row(run_dir: Path, metadata: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    choice = metadata["domain_svr_choice"]
    cv = pd.read_csv(run_dir / "inner_svr_cv.csv")
    eligible = cv.loc[
        cv["stage"].ne("preprocessing_screen")
        & cv["preprocessing"].eq(choice["preprocessing"])
        & np.isclose(cv["C"].astype(float), float(choice["C"]))
        & np.isclose(cv["gamma_factor"].astype(float), float(choice["gamma_factor"]))
        & np.isclose(cv["epsilon_z"].astype(float), float(choice["epsilon_z"]))
    ]
    if len(eligible) != 1:
        raise RuntimeError(f"Recorded SVR choice is not unique in {run_dir / 'inner_svr_cv.csv'}")
    return eligible.iloc[0], cv


def is_boundary(run_dir: Path, metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    choice = metadata["domain_svr_choice"]
    _, cv = selected_search_row(run_dir, metadata)
    cv = cv.loc[
        cv["stage"].ne("preprocessing_screen")
        & cv["preprocessing"].eq(choice["preprocessing"])
    ]
    axes: list[str] = []
    for parameter in ("C", "gamma_factor", "epsilon_z"):
        values = sorted(cv[parameter].astype(float).unique().tolist())
        if np.isclose(float(choice[parameter]), values[0]) or np.isclose(
            float(choice[parameter]), values[-1]
        ):
            axes.append(parameter)
    return bool(axes), axes


def collect_runs(root: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for path in sorted(root.rglob("metadata.json")):
        if path.parent.name.startswith("fold_"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            key = (str(metadata["trait"]), int(metadata["outer_fold"]))
            if key in result:
                raise RuntimeError(f"Duplicate candidate for {key}: {result[key]} and {path.parent}")
            result[key] = path.parent
    return result


def validate_wide_metadata(path: Path, metadata: dict[str, Any]) -> None:
    search = metadata.get("svr_search_space", {})
    hard = search.get("hard_limits", {}).get("epsilon_z", [None, None])
    if not hard or hard[-1] is None or float(hard[-1]) < 10.0:
        raise RuntimeError(f"Candidate is not the V25 wide-epsilon search: {path}")
    declared = {float(value) for value in search.get("epsilon_z", [])}
    missing = REQUIRED_WIDE_EPSILON_POINTS - declared
    if missing:
        raise RuntimeError(f"Candidate search is not a strict superset; missing {sorted(missing)}: {path}")
    if bool(metadata.get("test_labels_used_for_selection", True)):
        raise RuntimeError(f"Candidate metadata does not certify train-only selection: {path}")


def rebuild_aggregate(formal_dir: Path, manifest_path: Path, replacement_count: int, archive: Path) -> None:
    paths = sorted(formal_dir.glob("*/fold_*/predictions.parquet"))
    if len(paths) != 45:
        raise RuntimeError(f"Expected 45 final texture prediction files, found {len(paths)}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if len(predictions) != 43_677:
        raise RuntimeError(f"Expected 43,677 texture prediction rows, found {len(predictions)}")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("Final texture baseline has duplicate sample/target rows")
    predictions.to_parquet(formal_dir / "predictions.parquet", index=False, compression="zstd")
    metric_rows: list[dict[str, Any]] = []
    for (target, trait), frame in predictions.groupby(["target", "trait"], observed=True):
        for model, column in {
            "global_pls": "y_global_pls",
            "domain_pls": "y_domain_pls",
            "global_svr_selected_for_domain": "y_global_svr",
            "domain_svr": "y_domain_svr",
        }.items():
            metric_rows.append(
                {
                    "target": target,
                    "trait": trait,
                    "model": model,
                    **regression_metrics(frame["y_true"], frame[column]),
                }
            )
    pd.DataFrame(metric_rows).to_csv(formal_dir / "pooled_metrics.csv", index=False)
    manifest_path = manifest_path.resolve()
    summary = {
        "protocol": "V25 corrected-cohort non-overlapping five-fold nested baseline audit",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "targets_completed": int(predictions["target"].nunique()),
        "folds_completed": sorted(predictions["outer_fold"].unique().tolist()),
        "prediction_rows": int(len(predictions)),
        "test_labels_used_for_selection": False,
        "pls_component_grid": [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24],
        "svr_search_profile": "v25_staged",
        "wide_epsilon_replacements": replacement_count,
        "boundary_limited_archive": str(archive.resolve()),
    }
    (formal_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate strict-superset V25 SVR candidates, archive the boundary-limited folds, "
            "install the replacements and rebuild the aggregate baseline tables."
        )
    )
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    formal = args.formal_dir.resolve()
    candidates_root = args.candidate_dir.resolve()
    archive = args.archive_dir.resolve()
    audit_output = args.audit_output.resolve()
    for path in (formal, candidates_root, archive, audit_output.parent):
        if not within(path, project):
            raise RuntimeError(f"Path escaped project root: {path}")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {archive}")

    formal_runs = collect_runs(formal)
    if len(formal_runs) != 45:
        raise RuntimeError(f"Expected 45 formal folds before replacement, found {len(formal_runs)}")
    boundary: dict[tuple[str, int], tuple[Path, list[str]]] = {}
    for key, run_dir in formal_runs.items():
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        flagged, axes = is_boundary(run_dir, metadata)
        if flagged:
            boundary[key] = (run_dir, axes)

    candidates = collect_runs(candidates_root)
    if set(candidates) != set(boundary):
        raise RuntimeError(
            "Candidate keys must exactly match the formal boundary folds; "
            f"missing={sorted(set(boundary) - set(candidates))}, "
            f"extra={sorted(set(candidates) - set(boundary))}"
        )

    checks: list[dict[str, Any]] = []
    for key in sorted(boundary):
        old_dir, axes = boundary[key]
        new_dir = candidates[key]
        old_meta = json.loads((old_dir / "metadata.json").read_text(encoding="utf-8"))
        new_meta = json.loads((new_dir / "metadata.json").read_text(encoding="utf-8"))
        validate_wide_metadata(new_dir, new_meta)
        old_selected, _ = selected_search_row(old_dir, old_meta)
        new_selected, _ = selected_search_row(new_dir, new_meta)
        old_cv = float(old_selected["domain_cv_rmse"])
        new_cv = float(new_selected["domain_cv_rmse"])
        if new_cv > old_cv + max(1e-10, abs(old_cv) * 1e-10):
            raise RuntimeError(
                f"Strict-superset candidate has worse inner-CV RMSE for {key}: {new_cv} > {old_cv}"
            )
        old_pred = pd.read_parquet(old_dir / "predictions.parquet").sort_values("sample_id")
        new_pred = pd.read_parquet(new_dir / "predictions.parquet").sort_values("sample_id")
        if old_pred["sample_id"].tolist() != new_pred["sample_id"].tolist():
            raise RuntimeError(f"Test sample IDs differ for {key}")
        if not np.allclose(old_pred["y_true"], new_pred["y_true"], rtol=0, atol=1e-10):
            raise RuntimeError(f"Test truths differ for {key}")
        old_outer = regression_metrics(old_pred["y_true"], old_pred["y_domain_svr"])["rmse"]
        new_outer = regression_metrics(new_pred["y_true"], new_pred["y_domain_svr"])["rmse"]
        checks.append(
            {
                "trait": key[0],
                "outer_fold": key[1],
                "old_boundary_axes": ";".join(axes),
                "old_inner_cv_rmse": old_cv,
                "new_inner_cv_rmse": new_cv,
                "inner_cv_change_pct": 100.0 * (new_cv / old_cv - 1.0),
                "old_outer_rmse": old_outer,
                "new_outer_rmse": new_outer,
                "outer_rmse_change_pct": 100.0 * (new_outer / old_outer - 1.0),
                "old_prediction_sha256": sha256(old_dir / "predictions.parquet"),
                "new_prediction_sha256": sha256(new_dir / "predictions.parquet"),
                "sample_ids_and_truths_identical": True,
                "test_labels_used_for_selection": False,
            }
        )

    archive.mkdir(parents=True)
    for key in sorted(boundary):
        old_dir, _ = boundary[key]
        new_dir = candidates[key]
        archived = archive / key[0] / f"fold_{key[1]}"
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(archived))
        old_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(new_dir), str(old_dir))

    rebuild_aggregate(formal, args.manifest, len(checks), archive)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(audit_output, index=False)
    print(
        json.dumps(
            {
                "status": "PASS",
                "replacements": len(checks),
                "archive": str(archive),
                "audit": str(audit_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
