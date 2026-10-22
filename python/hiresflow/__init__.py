"""Minimal Python interface for the HiResFlow command-line runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class HiResFlowError(RuntimeError):
    """Raised when the HiResFlow executable fails."""


def _binary() -> str:
    configured = os.environ.get("HIRESFLOW_BIN")
    if configured:
        return configured
    repository_root = Path(__file__).resolve().parents[2]
    local_build = repository_root / "build" / "hiresflow"
    if local_build.is_file():
        return str(local_build)
    discovered = shutil.which("hiresflow")
    if discovered:
        return discovered
    raise HiResFlowError(
        "hiresflow executable not found; set HIRESFLOW_BIN or build the C++ project"
    )


def _execute(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        [_binary(), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise HiResFlowError(detail or f"hiresflow exited with {process.returncode}")
    return process


def segment(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    *,
    budget_mb: int = 6144,
    reserve_mb: int = 1024,
    batch_size: int | None = None,
    tile_sizes: tuple[int, ...] | None = None,
    overlap_ratio: float = 0.125,
    classes: int | None = None,
    input_size: tuple[int, int] | None = None,
    normalization: str = "imagenet",
    bgr: bool = False,
    calibration_path: str | Path | None = None,
    cpu: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run semantic segmentation and return the generated JSON report."""

    report = Path(report_path or f"{output_path}.report.json")
    arguments = [
        "run",
        "--backend",
        "onnx",
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--report",
        str(report),
        "--budget-mb",
        str(budget_mb),
        "--reserve-mb",
        str(reserve_mb),
        "--overlap-ratio",
        str(overlap_ratio),
    ]
    if batch_size is not None:
        arguments.extend(["--batch", str(batch_size)])
    if tile_sizes:
        arguments.extend(["--tile-size", ",".join(str(value) for value in tile_sizes)])
    if classes is not None:
        arguments.extend(["--classes", str(classes)])
    if input_size is not None:
        arguments.extend(
            ["--input-width", str(input_size[0]), "--input-height", str(input_size[1])]
        )
    arguments.extend(["--normalization", normalization])
    if bgr:
        arguments.append("--bgr")
    if calibration_path is not None:
        arguments.extend(["--calibration", str(calibration_path)])
    if cpu:
        arguments.append("--cpu")

    _execute(arguments)
    return json.loads(report.read_text(encoding="utf-8"))


def benchmark(
    input_path: str | Path,
    model_path: str | Path,
    *,
    report_path: str | Path,
    output_path: str | Path | None = None,
    warmup: int = 3,
    repeat: int = 10,
    budget_mb: int = 6144,
    reserve_mb: int = 1024,
    classes: int | None = None,
    input_size: tuple[int, int] | None = None,
    normalization: str = "imagenet",
    bgr: bool = False,
    calibration_path: str | Path | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Benchmark repeated end-to-end runs and return the JSON summary."""

    arguments = [
        "benchmark",
        "--backend",
        "onnx",
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--report",
        str(report_path),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--budget-mb",
        str(budget_mb),
        "--reserve-mb",
        str(reserve_mb),
    ]
    if output_path is not None:
        arguments.extend(["--output", str(output_path)])
    if classes is not None:
        arguments.extend(["--classes", str(classes)])
    if input_size is not None:
        arguments.extend(
            ["--input-width", str(input_size[0]), "--input-height", str(input_size[1])]
        )
    arguments.extend(["--normalization", normalization])
    if bgr:
        arguments.append("--bgr")
    if calibration_path is not None:
        arguments.extend(["--calibration", str(calibration_path)])
    if cpu:
        arguments.append("--cpu")

    _execute(arguments)
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def compare(
    input_path: str | Path,
    model_path: str | Path,
    *,
    output_prefix: str | Path,
    report_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    budget_mb: int = 6144,
    classes: int | None = None,
    input_size: tuple[int, int] | None = None,
    normalization: str = "imagenet",
    bgr: bool = False,
    calibration_path: str | Path | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Compare tiled inference with whole-image downscaling."""

    report = Path(report_path or f"{output_prefix}.report.json")
    arguments = [
        "compare",
        "--backend",
        "onnx",
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--output",
        str(output_prefix),
        "--report",
        str(report),
        "--budget-mb",
        str(budget_mb),
    ]
    if ground_truth_path is not None:
        arguments.extend(["--ground-truth", str(ground_truth_path)])
    if classes is not None:
        arguments.extend(["--classes", str(classes)])
    if input_size is not None:
        arguments.extend(
            ["--input-width", str(input_size[0]), "--input-height", str(input_size[1])]
        )
    arguments.extend(["--normalization", normalization])
    if bgr:
        arguments.append("--bgr")
    if calibration_path is not None:
        arguments.extend(["--calibration", str(calibration_path)])
    if cpu:
        arguments.append("--cpu")
    _execute(arguments)
    return json.loads(report.read_text(encoding="utf-8"))


def evaluate(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    *,
    classes: int,
    report_path: str | Path,
) -> dict[str, Any]:
    """Evaluate single-channel prediction and ground-truth index masks."""

    _execute(
        [
            "evaluate",
            "--prediction",
            str(prediction_path),
            "--ground-truth",
            str(ground_truth_path),
            "--classes",
            str(classes),
            "--report",
            str(report_path),
        ]
    )
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def run_directory(
    input_directory: str | Path,
    output_directory: str | Path,
    model_path: str | Path,
    *,
    budget_mb: int = 6144,
    classes: int | None = None,
    input_size: tuple[int, int] | None = None,
    normalization: str = "imagenet",
    bgr: bool = False,
    calibration_path: str | Path | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    """Process a directory and return its aggregate summary."""

    summary_path = Path(output_directory) / "summary.json"
    arguments = [
        "run-dir",
        "--backend",
        "onnx",
        "--model",
        str(model_path),
        "--input-dir",
        str(input_directory),
        "--output-dir",
        str(output_directory),
        "--report",
        str(summary_path),
        "--budget-mb",
        str(budget_mb),
    ]
    if classes is not None:
        arguments.extend(["--classes", str(classes)])
    if input_size is not None:
        arguments.extend(
            ["--input-width", str(input_size[0]), "--input-height", str(input_size[1])]
        )
    arguments.extend(["--normalization", normalization])
    if bgr:
        arguments.append("--bgr")
    if calibration_path is not None:
        arguments.extend(["--calibration", str(calibration_path)])
    if cpu:
        arguments.append("--cpu")
    _execute(arguments)
    return json.loads(summary_path.read_text(encoding="utf-8"))


__all__ = [
    "HiResFlowError",
    "benchmark",
    "compare",
    "evaluate",
    "run_directory",
    "segment",
]
