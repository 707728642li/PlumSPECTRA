"""Insert generated key tables into the supplementary Markdown placeholders."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/cea_final_revision/supplement_key_tables.md"
TARGET = ROOT / "manuscript/supplement_plumspectra_cea_v29.md"


def section(text: str, start: str, end: str | None) -> str:
    begin = text.index(start)
    finish = text.index(end, begin) if end else len(text)
    block = text[begin:finish].strip()
    lines = block.splitlines()
    return "\n".join(lines[2:]).strip()


def main() -> None:
    generated = SOURCE.read_text(encoding="utf-8")
    replacements = {
        "{{KEY_TABLE_S3}}": section(generated, "#### Table S3a", "#### Table S5"),
        "{{KEY_TABLE_S5}}": section(generated, "#### Table S5", "#### Table S10"),
        "{{KEY_TABLE_S10}}": section(generated, "#### Table S10", "#### Table S18"),
        "{{KEY_TABLE_S18}}": section(generated, "#### Table S18", "#### Table S41a"),
        "{{KEY_TABLE_S41}}": section(generated, "#### Table S41a", None),
    }
    target = TARGET.read_text(encoding="utf-8")
    for marker, block in replacements.items():
        if marker not in target:
            raise ValueError(f"Missing placeholder {marker}")
        target = target.replace(marker, block)
    TARGET.write_text(target, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
