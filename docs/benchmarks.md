# Benchmark log

All values below are measured on the project host, not estimated.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB
- Driver: 580.173.02
- ONNX Runtime: 1.31.0, CUDA 12 package
- Toolchain: CUDA 12.8
- Input: 1538x905 terminal screenshot (integration benchmark only)

## Initial CUDA smoke

Configuration: FCN ResNet-50, 512x512 input, 8 tiles, overlap 64, batch 8.

| Model | Provider | Inference | End-to-end | Peak device memory |
|---|---:|---:|---:|---:|
| INT8 | CUDA EP | 4.065 s | 4.416 s one shot | 1.269 GB |
| FP32 | CUDA EP | 0.693 s first run | 1.060 s first run | 2.345 GB |

The FP32 model placed 130 nodes on CUDA and 9 negligible nodes on CPU. CUDA
kernel time totaled 501 ms in the profiled run. The FP32 output matched the
CPU output exactly.

## Warm FP32 CUDA benchmark before merge optimization

3 warmup runs and 20 measured runs:

- End-to-end P50: 607.232 ms
- End-to-end P95: 609.454 ms
- Inference P50: 305.699 ms
- Mean preprocessing: 8.930 ms
- Mean merge: 264.552 ms
- Peak device memory used: 2349 MB

Merge time was 44% of end-to-end latency. This identified CPU-side class
decoding as the next optimization target.

## Merge optimization

The merge path now:

- Finds top-1 and top-2 logits in one pass.
- Uses a bounded top-margin confidence for overlap weighting.
- Avoids resampling when source and destination dimensions match.

On the CPU INT8 comparison path:

| Metric | Before | After |
|---|---:|---:|
| Mean merge | 401.646 ms | 96.727 ms |
| End-to-end P50 | 5190.058 ms | 4035.630 ms |

The optimized and previous outputs were byte-identical for this fixture.

## Optimized FP32 CUDA benchmark

3 warmup runs and 20 measured runs on the same 1538x905 input:

- End-to-end P50: 423.454 ms, down 30.3%
- End-to-end P95: 425.773 ms
- Inference P50: 297.035 ms
- Logits decode: 84.816 ms
- Output resize: 0.351 ms
- Fusion: 6.117 ms
- Total merge: 91.320 ms, down from 264.552 ms
- Peak device memory used: 2387 MB

## 4096x4096 batch sweep

Input: deterministic 4096x4096 performance fixture, 81 tiles at 512x512 with
64-pixel overlap. One warmup and two measured runs per batch.

| Batch | End-to-end P50 | Inference P50 | Peak device memory |
|---:|---:|---:|---:|
| 1 | 3324.540 ms | 2193.326 ms | 740 MiB |
| 2 | 4203.084 ms | 2904.383 ms | 1212 MiB |
| 4 | 4568.399 ms | 3250.450 ms | 2234 MiB |
| 8 | 4537.494 ms | 3176.669 ms | 2234 MiB |
| 16 | 4393.583 ms | 3025.455 ms | 7354 MiB |
| 32 | OOM | OOM | allocation failure |

For this FCN graph and ONNX Runtime CUDA build, batch 1 was the fastest
throughput configuration. Larger batches did not improve kernel utilization and
eventually exhausted the 8 GB device. The planner must therefore use measured
batch throughput rather than automatically choosing the largest batch that
passes a static memory estimate.

## Measured batch calibration

The calibration command selected batch 1 from the sweep. It rejected batch 32
before calibration through the conservative candidate list and separately
verified OOM recovery:

- Requested batch: 32
- ORT allocation failure: recognized
- Executed batch after fallback: 16
- Retries: 1
- Completed end-to-end: 5.613 s
- Peak device memory: 6330 MiB

An initial implementation produced `calibrated_sample_bytes=0` because ORT's
memory arena retained prior allocations between warm runs. The calibration
baseline was moved to the pre-inference, post-session-load point before the
fixed implementation was accepted.

## Calibrated planner

The corrected calibration selected batch 1 and measured a per-sample working
set of 474,480,640 bytes (452.5 MiB).

On the 1538x905 input:

| Metric | Uncalibrated batch 8 | Calibrated batch 1 |
|---|---:|---:|
| End-to-end P50 | 423.454 ms | 326.507 ms |
| End-to-end P95 | 425.773 ms | 337.497 ms |
| Inference P50 | 297.035 ms | 214.586 ms |
| Peak device memory | 2276 MiB | 702 MiB |

The calibrated plan reduced total latency by 22.9%, inference latency by 27.7%,
and peak memory by 69.2%.

## Real 6720x4480 sample comparison

Input: NASA Earth Day 2018, `KSC-20180418-PH_FWM01_0013`, 6720x4480.
Configuration: FP32 FCN, tile 512, overlap 64, calibrated batch 1, 150 tiles.

| Metric | Tiled | Downscaled to 512 |
|---|---:|---:|
| End-to-end | 6396.681 ms | 302.456 ms |
| Inference | 4262.973 ms | 25.243 ms |
| Merge/resize | 1542.220 ms | 234.176 ms |
| Semantic classes present | 18 | 4 |
| Dominant class fraction | 85.48% | 82.99% |

Pixel agreement between the two outputs was 93.66%. Tile-seam edge change rate
was 0.496%, compared with 0.188% on non-seam edges. The tiled output preserved
substantially more small-object classes, but the increased seam discontinuity
shows that fusion quality still needs dedicated evaluation with ground truth.
This sample demonstrates capability and failure modes; it is not yet an
accuracy benchmark.

### Overlap comparison

| Overlap | Tiles | Tiled end-to-end | Seam change | Non-seam change | Classes |
|---:|---:|---:|---:|---:|---:|
| 64 px | 150 | 6396.681 ms | 0.496% | 0.188% | 18 |
| 128 px | 216 | 8979.673 ms | 0.652% | 0.175% | 17 |

Doubling overlap increased runtime by 40.4%, did not reduce the measured seam
change rate, and removed one detected class. The default therefore remains at
64 pixels. This metric is a diagnostic, not proof that 64 pixels is optimal for
other models or scenes.

## 137-screenshot batch run

Input: 137 RGBA UI screenshots, 38 MiB total, sizes from 41x41 to 2552x1597.
The calibrated FP32 CUDA path processed 1145 tiles with one explicit warmup run.

- Images: 137 succeeded, 0 failed
- Processing wall time: 54.933 s
- Image throughput: 2.49 images/s
- Tile throughput: 20.84 tiles/s
- Per-image total latency P50/P95: 275.830 / 1150.209 ms
- Per-image inference P50/P95: 159.648 / 640.650 ms
- Mean preprocessing: 12.511 ms
- Mean merge: 91.358 ms
- Mean output write: 53.908 ms
- Correlation between tile count and total latency: 0.9984
- Batch retries: 0

Explicit warmup reduced the first image from 402.617 ms to 98.583 ms. The P95 is
dominated by 18 large screenshots with at least 17 tiles rather than random
runtime variance. For the 17+ tile bucket, mean inference was 601.550 ms, mean
merge was 243.636 ms, and mean output write was 156.589 ms.

These UI screenshots are a runtime stress test, not a semantic segmentation
quality benchmark. 60 outputs contained only one semantic class.
