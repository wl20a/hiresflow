#!/usr/bin/env python3
"""Generate a deterministic, license-free high-resolution performance fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_fixture(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    color_step = max(1, size // 17)
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 255) // max(1, size - 1),
                (y * 255) // max(1, size - 1),
                ((x // color_step + y // color_step) % 16) * 16,
            )

    draw = ImageDraw.Draw(image)
    for index in range(32):
        offset = index * size // 32
        width = max(8, size // 160)
        color = (255 - index * 7 % 255, index * 13 % 255, index * 29 % 255)
        draw.rectangle(
            [offset, size // 4, offset + width, 3 * size // 4],
            fill=color,
        )
        draw.ellipse(
            [size // 3, offset, 2 * size // 3, offset + width],
            fill=(color[2], color[0], color[1]),
        )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/samples/perf-4096x4096.png"),
    )
    args = parser.parse_args()
    if args.size < 512:
        raise SystemExit("--size must be at least 512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_fixture(args.size).save(args.output, optimize=False)
    print(args.output)


if __name__ == "__main__":
    main()
