from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def natural_page(path: Path) -> int:
    match = re.search(r"page-(\d+)\.png$", path.name)
    return int(match.group(1)) if match else 10**9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=4)
    args = parser.parse_args()
    pages = sorted(args.input_dir.glob("page-*.png"), key=natural_page)
    if not pages:
        raise FileNotFoundError(f"No rendered pages in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = 800, 1040
    font = ImageFont.load_default(size=18)
    for start in range(0, len(pages), args.per_sheet):
        selected = pages[start:start + args.per_sheet]
        sheet = Image.new("RGB", (cell_width * 2, cell_height * 2), "#D8DCE3")
        draw = ImageDraw.Draw(sheet)
        for offset, page in enumerate(selected):
            with Image.open(page) as opened:
                image = opened.convert("RGB")
                image.thumbnail((cell_width - 24, cell_height - 48), Image.Resampling.LANCZOS)
                x = (offset % 2) * cell_width + (cell_width - image.width) // 2
                y = (offset // 2) * cell_height + 34
                sheet.paste(image, (x, y))
                draw.text((offset % 2 * cell_width + 12, offset // 2 * cell_height + 8),
                          f"Page {natural_page(page)}", fill="#17243B", font=font)
        first = natural_page(selected[0])
        last = natural_page(selected[-1])
        sheet.save(args.output_dir / f"pages_{first:02d}_{last:02d}.png", optimize=True)
    print(f"Created {((len(pages) - 1) // args.per_sheet) + 1} contact sheets for {len(pages)} pages")


if __name__ == "__main__":
    main()
