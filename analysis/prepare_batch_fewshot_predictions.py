from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    required = {"sample_id", "cultivar_ascii", "heldout_batch", "target", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.rename(columns={"cultivar_ascii": "source_cultivar_ascii"})
    frame["cultivar_ascii"] = frame["heldout_batch"]
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output.resolve(), index=False, compression="zstd")


if __name__ == "__main__":
    main()
