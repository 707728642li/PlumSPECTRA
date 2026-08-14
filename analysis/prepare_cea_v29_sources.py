"""Branch the current V28 evidence-enhanced sources into the CEA V29 release."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (
        ROOT / "manuscript" / "manuscript_plumspectra_v28.md",
        ROOT / "manuscript" / "manuscript_plumspectra_cea_v29.md",
    ),
    (
        ROOT / "manuscript" / "supplement_plumspectra_v28.md",
        ROOT / "manuscript" / "supplement_plumspectra_cea_v29.md",
    ),
)


def main() -> None:
    for source, target in PAIRS:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing CEA source: {target}")
        target.write_text(source.read_text(encoding="utf-8-sig"), encoding="utf-8", newline="\n")
        print(f"created {target.relative_to(ROOT)} from {source.name}")


if __name__ == "__main__":
    main()
