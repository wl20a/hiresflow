# v0.1 progress

Updated: 2026-10-22

## Implemented

- C++17 core library and CLI.
- PNG/JPG decoding and colored or 8-bit index mask output.
- ONNX Runtime 1.31 C++ backend with CUDA provider preference and CPU fallback.
- Model inspection for input size, output classes, and fixed batch dimensions.
- VRAM-budget batch planning with a configurable safety reserve.
- Edge-anchored tile generation with overlap.
- Bounded-memory weighted overlap fusion.
- Runtime OOM detection and batch halving.
- Measured batch calibration with reusable model-size-bound config.
- Single-image reports with stage timings and adaptive batch fields.
- Repeated P50/P95 benchmark reports.
- Whole-image downscaling baseline and consistency diagnostics.
- Optional single-channel index ground truth, mIoU, and pixel accuracy.
- Single-image and directory-level execution.
- Minimal Python wrapper around the C++ CLI.
- Unit tests for coverage, fixed batch constraints, OOM fallback, fusion,
  coordinate-preserving output, index mask round trips, and metrics.

## Verified in the Codex environment

- Release build succeeds with ONNX Runtime CUDA 12 package.
- Unit tests pass.
- Synthetic end-to-end path works on a 1538x905 image.
- Public FCN ResNet-50 FP32 ONNX model completes on the CPU provider.
- The FP32 model passes through the RTX 5070 CUDA provider.
- FP32 CUDA warm-run P50 is 423 ms after merge optimization.
- A 4096x4096 sweep found batch 1 fastest for the current FCN graph.
- Measured calibration selected batch 1, reduced small-input P50 to 327 ms,
  and reduced peak memory from 2276 MiB to 702 MiB.
- A 6720x4480 NASA sample completed with calibrated tiling. Tiled output
  contained 18 classes versus 4 for whole-image downscaling; 93.66% of pixels
  agreed.
- Doubling overlap from 64 to 128 pixels increased runtime by 40.4% without
  reducing the measured seam change rate; 64 pixels remains the default.
- A 137-image screenshot directory completed with 0 failures, 1145 tiles, and
  2.49 images/s throughput on the calibrated CUDA path.
- Benchmark and comparison reports are valid JSON.
- Python API calls complete through the C++ executable.

The CPU numbers are correctness and integration checks, not performance claims.

## Host validation

The Codex process does not receive `/dev/nvidia*`, so GPU runs are executed
from the normal host terminal. Use the host scripts for calibration, profiling,
benchmarking, and batch validation:

```bash
bash scripts/host_gpu_smoke.sh /path/to/large-image.jpg
```

## Not yet implemented

- Multi-Stream execution.
- Automatic FP16 model selection.
- GPU decode/preprocessing with nvImageCodec or CV-CUDA.
- Dataset slicing with synchronized mask, box, and keypoint transforms.
- Full probability fusion for all classes at original resolution.
- A ground-truth annotated high-resolution benchmark dataset.
