"""Reorder the CEA manuscript to the conventional IMRaD sequence.

The operation is assertion guarded and may be run once on the V29 source.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manuscript" / "manuscript_plumspectra_cea_v29.md"
ORDER = [
    "Abstract",
    "Introduction",
    "Materials and methods",
    "Results",
    "Discussion",
    "Conclusions",
    "Data availability",
    "Acknowledgments",
    "Contributions",
    "Conflict of interests",
    "Supplementary information",
    "Figure legends",
    "References",
]


def main() -> None:
    text = PATH.read_text(encoding="utf-8-sig")
    first = text.index("\n## Abstract\n")
    front = text[:first].rstrip()
    sections: dict[str, str] = {}
    starts: list[tuple[int, str]] = []
    for name in ORDER:
        marker = f"\n## {name}\n"
        count = text.count(marker)
        if count != 1:
            raise AssertionError(f"expected one {marker!r}, found {count}")
        starts.append((text.index(marker), name))
    starts.sort()
    for i, (start, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections[name] = text[start:end].strip()
    rebuilt = front + "\n\n" + "\n\n".join(sections[name] for name in ORDER) + "\n"
    PATH.write_text(rebuilt, encoding="utf-8", newline="\n")
    print("reordered CEA V29 sections: " + " -> ".join(ORDER))


if __name__ == "__main__":
    main()
