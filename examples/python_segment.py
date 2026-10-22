#!/usr/bin/env python3
"""Run one image through the HiResFlow Python wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hiresflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = hiresflow.segment(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        input_size=(512, 512),
        classes=21,
        calibration_path=args.calibration,
        report_path=args.report,
    )
    print(json.dumps(report["timing_ms"], indent=2))


if __name__ == "__main__":
    main()
