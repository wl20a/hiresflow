# Examples

## CLI

```bash
./build/hiresflow run \
  --backend onnx \
  --model work/models/fcn-resnet50-12.onnx \
  --calibration outputs/host-gpu-calibration.conf \
  --input path/to/image.jpg \
  --output outputs/mask.png \
  --index-output outputs/mask.index.png \
  --report outputs/report.json \
  --input-width 512 \
  --input-height 512 \
  --classes 21
```

## Python

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
export HIRESFLOW_BIN="$PWD/build/hiresflow"

python examples/python_segment.py \
  --input path/to/image.jpg \
  --output outputs/mask.png \
  --model work/models/fcn-resnet50-12.onnx \
  --calibration outputs/host-gpu-calibration.conf
```

The Python package initially wraps the C++ executable. A native extension
module is not yet part of the v0.1 preview.
