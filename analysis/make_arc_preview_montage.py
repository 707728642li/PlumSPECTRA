from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.bmp"), key=lambda p: p.name.lower())
    if not files:
        raise SystemExit("No BMP previews found")
    images = [Image.open(file).convert("RGB") for file in files]
    cell_w = max(image.width for image in images) * args.scale
    image_h = max(image.height for image in images) * args.scale
    label_h = 28
    cell_h = image_h + label_h
    rows = math.ceil(len(images) / args.columns)
    canvas = Image.new("RGB", (cell_w * args.columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for index, (file, image) in enumerate(zip(files, images)):
        row, col = divmod(index, args.columns)
        x, y = col * cell_w, row * cell_h
        batch = file.name.split("__", 1)[0]
        draw.text((x + 5, y + 4), batch, fill="black", font=font)
        resized = image.resize((image.width * args.scale, image.height * args.scale), Image.Resampling.NEAREST)
        canvas.paste(resized, (x, y + label_h))
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_png)
    print(args.output_png)


if __name__ == "__main__":
    main()

