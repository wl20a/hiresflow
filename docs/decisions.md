# Architecture decisions

## ADR-001: Keep geometry in the runtime core

Tile generation, overlap weights, and original-coordinate mapping are core
runtime responsibilities. A model backend receives normalized NCHW tensors and
returns CHW class scores. It does not decide where a prediction belongs in the
source image.

This separation makes coordinate tests independent from ONNX Runtime and
allows another execution provider to be added without changing stitching.

## ADR-002: Treat the model batch dimension as a planning constraint

The first ONNX smoke model exposed a fixed batch dimension of one. Budget-based
planning alone therefore selected an invalid batch size. Model metadata now
constrains the batch search before the memory estimate is applied.

## ADR-003: Do not claim an exact VRAM predictor

Transformer and CNN activation memory cannot be derived reliably from model file
size. The preview uses a conservative static estimate and accepts externally
calibrated per-sample bytes. Runtime pressure sampling is separate from the
initial plan.

## ADR-004: Single-Stream first

The executor interface is deliberately simple for the preview. Multi-Stream
execution will be introduced only after the single-Stream path has stable
coordinates, deterministic output, and measurable stage timings.

## ADR-005: Winner fusion with spatial weights

Full class-probability accumulation for every source pixel is too large for
multi-class segmentation at 4K and above. The preview keeps four class
candidates per pixel and accumulates confidence multiplied by a tile-edge
weight. Confidence is the top-1 versus top-2 logit margin passed through a
logistic function; it is not a calibrated softmax probability. This is
bounded-memory overlap fusion, not full probability averaging.

## ADR-006: Ground truth requires explicit class-index masks

Reading a palette PNG as RGB and guessing how colors map to classes creates
silent evaluation errors. The evaluator therefore accepts only single-channel
index masks. Colored masks remain visualization artifacts. Dataset-specific
palette mappings can be added later as explicit configuration.
