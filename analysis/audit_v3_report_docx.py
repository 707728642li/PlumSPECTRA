from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def attr(node: ET.Element | None, name: str) -> str | None:
    return None if node is None else node.get(W + name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the narrative-proposal V3 DOCX token map.")
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with zipfile.ZipFile(args.docx) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        styles = ET.fromstring(archive.read("word/styles.xml"))

    text = "".join(node.text or "" for node in document.iter(W + "t"))
    section = document.find(f".//{W}sectPr")
    page_size = section.find(W + "pgSz")
    margins = section.find(W + "pgMar")
    geometry = {
        "page_width": attr(page_size, "w"),
        "page_height": attr(page_size, "h"),
        "margin_top": attr(margins, "top"),
        "margin_right": attr(margins, "right"),
        "margin_bottom": attr(margins, "bottom"),
        "margin_left": attr(margins, "left"),
        "header_distance": attr(margins, "header"),
        "footer_distance": attr(margins, "footer"),
    }
    expected_geometry = {
        "page_width": "12240",
        "page_height": "15840",
        "margin_top": "1440",
        "margin_right": "1440",
        "margin_bottom": "1440",
        "margin_left": "1440",
        "header_distance": "708",
        "footer_distance": "708",
    }

    normal = next(
        style for style in styles.findall(W + "style") if style.get(W + "styleId") == "Normal"
    )
    rpr = normal.find(W + "rPr")
    ppr = normal.find(W + "pPr")
    spacing = ppr.find(W + "spacing")
    normal_details = {
        "font": attr(rpr.find(W + "rFonts"), "ascii"),
        "size_half_points": attr(rpr.find(W + "sz"), "val"),
        "alignment": attr(ppr.find(W + "jc"), "val"),
        "after_twips": attr(spacing, "after"),
        "line_twips": attr(spacing, "line"),
        "line_rule": attr(spacing, "lineRule"),
    }
    expected_normal = {
        "font": "Calibri",
        "size_half_points": "22",
        "alignment": "both",
        "after_twips": "160",
        "line_twips": "320",
        "line_rule": "auto",
    }

    table_reports: list[dict[str, object]] = []
    for index, table in enumerate(document.findall(f".//{W}tbl"), start=1):
        properties = table.find(W + "tblPr")
        grid_widths = [int(attr(node, "w") or 0) for node in table.find(W + "tblGrid").findall(W + "gridCol")]
        cells_match = True
        for row in table.findall(W + "tr"):
            widths = [int(attr(cell.find(f"{W}tcPr/{W}tcW"), "w") or 0) for cell in row.findall(W + "tc")]
            if widths != grid_widths:
                cells_match = False
                break
        first_row = table.find(W + "tr")
        header_marked = first_row.find(f"{W}trPr/{W}tblHeader") is not None
        record = {
            "table": index,
            "width": attr(properties.find(W + "tblW"), "w"),
            "indent": attr(properties.find(W + "tblInd"), "w"),
            "grid_sum": sum(grid_widths),
            "cell_widths_match_grid": cells_match,
            "header_row_marked": header_marked,
        }
        record["pass"] = bool(
            record["width"] == "9360"
            and record["indent"] == "120"
            and record["grid_sum"] == 9360
            and cells_match
            and header_marked
        )
        table_reports.append(record)

    heading_text = [
        "".join(node.text or "" for node in paragraph.iter(W + "t"))
        for paragraph in document.findall(f".//{W}p")
        if paragraph.find(f"{W}pPr/{W}pStyle") is not None
        and attr(paragraph.find(f"{W}pPr/{W}pStyle"), "val") in {"Heading1", "Heading2", "Heading3"}
    ]
    exact_heights = [
        node
        for node in document.findall(f".//{W}trHeight")
        if attr(node, "hRule") == "exact"
    ]
    checks = {
        "narrative_proposal_page_geometry": geometry == expected_geometry,
        "narrative_proposal_normal_style": normal_details == expected_normal,
        "five_data_tables": len(table_reports) == 5,
        "fixed_dxa_table_geometry_and_headers": all(record["pass"] for record in table_reports),
        "one_figure": len(document.findall(f".//{A}blip")) == 1,
        "no_exact_row_heights": not exact_heights,
        "fifteen_numbered_sections_plus_title": len(heading_text) == 16,
        "no_unresolved_tokens": "{{" not in text and "}}" not in text,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "docx": str(args.docx.resolve()),
        "preset": "narrative_proposal with editorial_cover",
        "checks": checks,
        "page_geometry": geometry,
        "normal_style": normal_details,
        "tables": table_reports,
        "headings": heading_text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
