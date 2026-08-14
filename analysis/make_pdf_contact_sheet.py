#!/usr/bin/env python3
"""Create a labelled contact sheet from rendered PDF page PNGs for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=270)
    args = parser.parse_args()

    pages = sorted(args.page_dir.glob("*.png"))
    if not pages:
        raise SystemExit(f"No PNG pages found in {args.page_dir}")

    font = ImageFont.load_default(size=18)
    label_height = 28
    gap = 14
    thumbnails: list[tuple[Path, Image.Image]] = []
    for page in pages:
        with Image.open(page) as source:
            ratio = args.thumb_width / source.width
            thumb = source.convert("RGB").resize(
                (args.thumb_width, round(source.height * ratio)), Image.Resampling.LANCZOS
            )
        thumbnails.append((page, thumb))

    cell_width = args.thumb_width + gap
    cell_height = max(image.height for _, image in thumbnails) + label_height + gap
    rows = (len(thumbnails) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * cell_width + gap, rows * cell_height + gap), "#D9DDE3")
    draw = ImageDraw.Draw(sheet)
    for index, (path, thumb) in enumerate(thumbnails):
        row, column = divmod(index, args.columns)
        x = gap + column * cell_width
        y = gap + row * cell_height
        sheet.paste(thumb, (x, y + label_height))
        draw.text((x, y + 3), path.stem, fill="#17233B", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
