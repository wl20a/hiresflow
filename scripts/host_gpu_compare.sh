#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-}"
if [[ -z "${image_path}" ]]; then
  echo "usage: $0 /path/to/large-image.(jpg|png) [ground-truth-index.png]" >&2
  exit 2
fi
ground_truth="${2:-}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${HIRESFLOW_BIN:-${repository_root}/build/hiresflow}"
model="${HIRESFLOW_MODEL:-${repository_root}/work/models/fcn-resnet50-12.onnx}"
prefix="${HIRESFLOW_REPORT_PREFIX:-${repository_root}/outputs/host-gpu-comparison}"
calibration="${HIRESFLOW_CALIBRATION:-}"
budget_mb="${HIRESFLOW_BUDGET_MB:-}"
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

arguments=(
  compare
  --backend onnx
  --model "${model}"
  --input "${image_path}"
  --output "${prefix}"
  --report "${prefix}.report.json"
  --input-width 512
  --input-height 512
  --classes 21
  --reserve-mb 1024
  --overlap-ratio "${overlap_ratio}"
)
if [[ -n "${ground_truth}" ]]; then
  arguments+=(--ground-truth "${ground_truth}")
fi
if [[ -n "${calibration}" ]]; then
  arguments+=(--calibration "${calibration}")
fi
if [[ -n "${budget_mb}" ]]; then
  arguments+=(--budget-mb "${budget_mb}")
fi

mkdir -p "$(dirname "${prefix}")"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"${binary}" "${arguments[@]}"
