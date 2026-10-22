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
report="${HIRESFLOW_REPORT:-${repository_root}/outputs/host-gpu-benchmark.json}"
warmup="${HIRESFLOW_WARMUP:-3}"
repeat="${HIRESFLOW_REPEAT:-20}"
budget_mb="${HIRESFLOW_BUDGET_MB:-}"
batch="${HIRESFLOW_BATCH:-}"
calibration="${HIRESFLOW_CALIBRATION:-}"
overlap_ratio="${HIRESFLOW_OVERLAP:-0.125}"
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

mkdir -p "$(dirname "${report}")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

arguments=(
  benchmark
  --backend onnx
  --model "${model}"
  --input "${image_path}"
  --report "${report}"
  --output "${repository_root}/outputs/host-gpu-benchmark.mask.png"
  --input-width 512
  --input-height 512
  --classes 21
  --reserve-mb 1024
  --overlap-ratio "${overlap_ratio}"
  --warmup "${warmup}"
  --repeat "${repeat}"
)
if [[ -n "${budget_mb}" ]]; then
  arguments+=(--budget-mb "${budget_mb}")
fi
if [[ -n "${batch}" ]]; then
  arguments+=(--batch "${batch}")
fi
if [[ -n "${calibration}" ]]; then
  arguments+=(--calibration "${calibration}")
fi

"${binary}" "${arguments[@]}"
