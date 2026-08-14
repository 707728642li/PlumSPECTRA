"""Create the V28 manuscript sources from the frozen V27 release.

This is deliberately a versioning-only step: all scientific edits are applied
to the resulting V28 files, while the audited V27 sources remain immutable.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (
        ROOT / "manuscript" / "manuscript_plumspectra_v27.md",
        ROOT / "manuscript" / "manuscript_plumspectra_v28.md",
    ),
    (
        ROOT / "manuscript" / "supplement_plumspectra_v27.md",
        ROOT / "manuscript" / "supplement_plumspectra_v28.md",
    ),
)


def main() -> None:
    for source, target in PAIRS:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing V28 source: {target}")
        text = source.read_text(encoding="utf-8-sig")
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"created {target.relative_to(ROOT)} from {source.name}")


if __name__ == "__main__":
    main()
