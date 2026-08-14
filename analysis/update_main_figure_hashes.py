"""Record hashes for the current publication PDF and PNG main figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results/v26_claudecode_integration/figures_integrated"
OUT = ROOT / "results/v29_cea_submission/frozen_main_figure_sha256.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for number in range(1, 7):
        for suffix in ("pdf", "png"):
            relative = Path(f"results/v26_claudecode_integration/figures_integrated/Figure_{number}_v26.{suffix}")
            path = ROOT / relative
            files[relative.as_posix()] = sha256(path)
    payload = {
        "purpose": "Integrity record for the current main-figure PDF and PNG files; Figures 4–6 include targeted review corrections",
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
