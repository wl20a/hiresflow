#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"
mkdir -p outputs

bash scripts/fetch_deps.sh
bash scripts/fetch_model.sh
bash scripts/fetch_sample.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHIRESFLOW_BUILD_TESTS=ON
cmake --build build -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
ctest --test-dir build --output-on-failure

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  bash scripts/host_gpu_calibrate.sh work/samples/perf-4096x4096.png
  bash scripts/host_gpu_smoke.sh work/samples/nasa/earth-day-2018.jpg
  echo "GPU quickstart complete."
else
  ./build/hiresflow run \
    --backend synthetic \
    --input work/samples/perf-4096x4096.png \
    --output outputs/quickstart-synthetic.png \
    --input-width 256 \
    --budget-mb 6144 \
    --report outputs/quickstart-synthetic.report.json
  echo "Build and synthetic runtime check complete."
  echo "No accessible NVIDIA device was detected; pass --cpu when using ONNX Runtime."
fi
