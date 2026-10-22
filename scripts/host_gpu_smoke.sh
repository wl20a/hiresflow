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
budget_mb="${HIRESFLOW_BUDGET_MB:-}"
calibration="${HIRESFLOW_CALIBRATION:-}"
if [[ -z "${calibration}" && -f "${repository_root}/outputs/host-gpu-calibration.conf" ]]; then
  calibration="${repository_root}/outputs/host-gpu-calibration.conf"
fi

if [[ ! -x "${binary}" ]]; then
  echo "hiresflow binary not found at ${binary}" >&2
  exit 2
fi
if [[ ! -f "${model}" ]]; then
  echo "model not found at ${model}; run scripts/fetch_model.sh first" >&2
  exit 2
fi

mkdir -p "${repository_root}/outputs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

arguments=(
  run
  --backend onnx
  --model "${model}"
  --input "${image_path}"
  --output "${repository_root}/outputs/host-gpu-demo.png"
  --report "${repository_root}/outputs/host-gpu-demo.report.json"
  --input-width 512
  --input-height 512
  --classes 21
  --reserve-mb 1024
  --overlap-ratio 0.125
)
if [[ -n "${budget_mb}" ]]; then
  arguments+=(--budget-mb "${budget_mb}")
fi
if [[ "${HIRESFLOW_PROFILE:-0}" == "1" ]]; then
  arguments+=(--profile)
fi
if [[ -n "${calibration}" ]]; then
  arguments+=(--calibration "${calibration}")
fi

"${binary}" "${arguments[@]}"
