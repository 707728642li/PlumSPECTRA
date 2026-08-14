from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw


def page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    pages = sorted(args.input_dir.glob("page-*.png"), key=page_number)
    if not pages:
        raise SystemExit("No page PNGs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pair_index in range(0, len(pages), 2):
        pair = pages[pair_index:pair_index + 2]
        images = [Image.open(path).convert("RGB") for path in pair]
        gap = 30
        width = sum(image.width for image in images) + gap * (len(images) - 1)
        height = max(image.height for image in images)
        canvas = Image.new("RGB", (width, height), "#D9DEE6")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for path, page in zip(pair, images, strict=True):
            canvas.paste(page, (x, 0))
            draw.rectangle((x + 8, 8, x + 105, 42), fill="#17243B")
            draw.text((x + 18, 16), f"PAGE {page_number(path)}", fill="white")
            x += page.width + gap
        first = page_number(pair[0])
        last = page_number(pair[-1])
        output = args.output_dir / f"sheet_{pair_index // 2 + 1:02d}_pages_{first:02d}_{last:02d}.png"
        canvas.save(output, optimize=True)
        print(output)


if __name__ == "__main__":
    main()
