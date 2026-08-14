from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def attr(node: ET.Element | None, name: str) -> str | None:
    return None if node is None else node.get(W + name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--expected-tables", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
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
    geometry_pass = geometry == {
        "page_width": "12240",
        "page_height": "15840",
        "margin_top": "1440",
        "margin_right": "1440",
        "margin_bottom": "1440",
        "margin_left": "1440",
        "header_distance": "648",
        "footer_distance": "648",
    }

    normal = next(
        style for style in styles.findall(W + "style") if style.get(W + "styleId") == "Normal"
    )
    normal_rpr = normal.find(W + "rPr")
    normal_ppr = normal.find(W + "pPr")
    spacing = normal_ppr.find(W + "spacing")
    normal_details = {
        "font": attr(normal_rpr.find(W + "rFonts"), "ascii"),
        "size_half_points": attr(normal_rpr.find(W + "sz"), "val"),
        "alignment": attr(normal_ppr.find(W + "jc"), "val"),
        "after_twips": attr(spacing, "after"),
        "line_twips": attr(spacing, "line"),
        "line_rule": attr(spacing, "lineRule"),
    }
    normal_pass = (
        normal_details["font"] == "Calibri"
        and normal_details["size_half_points"] == "22"
        and normal_details["alignment"] == "both"
        and normal_details["after_twips"] == "160"
        and normal_details["line_rule"] == "auto"
    )

    table_reports = []
    for index, table in enumerate(document.findall(f".//{W}tbl"), start=1):
        properties = table.find(W + "tblPr")
        width = attr(properties.find(W + "tblW"), "w")
        indent = attr(properties.find(W + "tblInd"), "w")
        grid_widths = [int(attr(node, "w") or 0) for node in table.find(W + "tblGrid").findall(W + "gridCol")]
        cell_widths_ok = True
        for row in table.findall(W + "tr"):
            cells = row.findall(W + "tc")
            if len(cells) != len(grid_widths):
                cell_widths_ok = False
                break
            widths = [int(attr(cell.find(f"{W}tcPr/{W}tcW"), "w") or 0) for cell in cells]
            if widths != grid_widths:
                cell_widths_ok = False
                break
        table_reports.append(
            {
                "table": index,
                "width": width,
                "indent": indent,
                "grid_widths": grid_widths,
                "grid_sum": sum(grid_widths),
                "cell_widths_match_grid": cell_widths_ok,
                "pass": width == "9360" and indent == "120" and sum(grid_widths) == 9360 and cell_widths_ok,
            }
        )

    image_count = len(document.findall(f".//{A}blip"))
    headings = [
        "".join(node.text or "" for node in paragraph.iter(W + "t"))
        for paragraph in document.findall(f".//{W}p")
        if (paragraph.find(f"{W}pPr/{W}pStyle") is not None)
        and attr(paragraph.find(f"{W}pPr/{W}pStyle"), "val") in {"Heading1", "Heading2", "Heading3"}
    ]
    fixed_heights = [
        attr(node, "hRule")
        for node in document.findall(f".//{W}trHeight")
        if attr(node, "hRule") == "exact"
    ]
    placeholders = sorted(set(re.findall(r"\[[A-Z][A-Z0-9 /,._°-]{3,}\]", text)))
    checks = {
        "page_geometry": geometry_pass,
        "normal_style": normal_pass,
        "table_count": len(table_reports) == args.expected_tables,
        "table_geometry": all(record["pass"] for record in table_reports),
        "image_count": image_count == args.expected_images,
        "no_exact_row_heights": not fixed_heights,
        "no_unresolved_template_tokens": "{{" not in text,
        "headings_present": bool(headings),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "docx": str(args.docx.resolve()),
        "checks": checks,
        "page_geometry": geometry,
        "normal_style": normal_details,
        "tables": table_reports,
        "images": image_count,
        "headings": headings,
        "submission_placeholders": placeholders,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
