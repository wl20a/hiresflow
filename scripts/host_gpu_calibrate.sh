#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-}"
if [[ -z "${image_path}" ]]; then
  echo "usage: $0 /path/to/large-image.(jpg|png)" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${HIRESFLOW_BIN:-${repository_root}/build/hiresflow}"
model="${HIRESFLOW_MODEL:-${repository_root}/work/models/fcn-resnet50-12.onnx}"
report="${HIRESFLOW_REPORT:-${repository_root}/outputs/host-gpu-calibration.json}"
config="${HIRESFLOW_CONFIG:-${repository_root}/outputs/host-gpu-calibration.conf}"
batches="${HIRESFLOW_BATCHES:-1,2,4,8,16}"
warmup="${HIRESFLOW_WARMUP:-1}"
repeat="${HIRESFLOW_REPEAT:-2}"
budget_mb="${HIRESFLOW_BUDGET_MB:-}"

if [[ ! -x "${binary}" ]]; then
  echo "hiresflow binary not found at ${binary}" >&2
  exit 2
fi
if [[ ! -f "${model}" ]]; then
  echo "model not found at ${model}; run scripts/fetch_model.sh first" >&2
  exit 2
fi

mkdir -p "$(dirname "${report}")" "$(dirname "${config}")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

arguments=(
  calibrate
  --backend onnx
  --model "${model}"
  --input "${image_path}"
  --batches "${batches}"
  --warmup "${warmup}"
  --repeat "${repeat}"
  --report "${report}"
  --config "${config}"
  --input-width 512
  --input-height 512
  --classes 21
  --reserve-mb 1024
  --overlap-ratio 0.125
)
if [[ -n "${budget_mb}" ]]; then
  arguments+=(--budget-mb "${budget_mb}")
fi

"${binary}" "${arguments[@]}"
