#!/usr/bin/env python3
"""Render a side-by-side tiled/baseline comparison with a disagreement overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_label_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode not in {"L", "I", "I;16"}:
        raise SystemExit(f"expected a single-channel index mask: {path}")
    return image


def add_panel_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width // 70))
    margin = max(8, image.width // 180)
    box = draw.textbbox((margin, margin), text, font=font)
    draw.rounded_rectangle(
        [box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin],
        radius=margin,
        fill=(0, 0, 0),
    )
    draw.text((margin, margin), text, font=font, fill=(255, 255, 255))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tiled", type=Path, required=True)
    parser.add_argument("--downscaled", type=Path, required=True)
    parser.add_argument("--tiled-index", type=Path, required=True)
    parser.add_argument("--downscaled-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    source = Image.open(args.input).convert("RGB")
    tiled = Image.open(args.tiled).convert("RGB")
    downscaled = Image.open(args.downscaled).convert("RGB")
    tiled_index = load_label_image(args.tiled_index)
    downscaled_index = load_label_image(args.downscaled_index)
    if len({source.size, tiled.size, downscaled.size, tiled_index.size, downscaled_index.size}) != 1:
        raise SystemExit("all inputs must have the same dimensions")

    width = min(args.width, source.width)
    height = round(source.height * width / source.width)
    source = source.resize((width, height), Image.Resampling.LANCZOS)
    tiled = tiled.resize((width, height), Image.Resampling.NEAREST)
    downscaled = downscaled.resize((width, height), Image.Resampling.NEAREST)
    tiled_index = tiled_index.resize((width, height), Image.Resampling.NEAREST)
    downscaled_index = downscaled_index.resize((width, height), Image.Resampling.NEAREST)

    tiled_pixels = list(tiled_index.getdata())
    downscaled_pixels = list(downscaled_index.getdata())
    difference = Image.new("L", (width, height))
    difference.putdata(
        [255 if left != right else 0 for left, right in zip(tiled_pixels, downscaled_pixels)]
    )
    overlay = source.copy()
    red = Image.new("RGB", (width, height), (255, 32, 32))
    overlay.paste(red, mask=difference.point(lambda value: 170 if value else 0))

    add_panel_label(source, "input")
    add_panel_label(tiled, "tiled 512 / overlap 64")
    add_panel_label(downscaled, "whole image -> 512")
    add_panel_label(overlay, "label disagreement")

    canvas = Image.new("RGB", (width * 4, height), (32, 32, 32))
    for index, panel in enumerate((source, tiled, downscaled, overlay)):
        canvas.paste(panel, (index * width, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
