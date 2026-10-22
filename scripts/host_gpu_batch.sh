#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:-}"
if [[ -z "${input_dir}" ]]; then
  echo "usage: $0 /path/to/image-directory [output-directory]" >&2
  exit 2
fi
if [[ ! -d "${input_dir}" ]]; then
  echo "input directory does not exist: ${input_dir}" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${2:-${HIRESFLOW_OUTPUT_DIR:-${repository_root}/outputs/screenshot-batch}}"
binary="${HIRESFLOW_BIN:-${repository_root}/build/hiresflow}"
model="${HIRESFLOW_MODEL:-${repository_root}/work/models/fcn-resnet50-12.onnx}"
calibration="${HIRESFLOW_CALIBRATION:-}"
overlap_ratio="${HIRESFLOW_OVERLAP:-0.125}"
warmup="${HIRESFLOW_WARMUP:-1}"
summary_path="${output_dir}/summary.json"

if [[ -z "${calibration}" && -f "${repository_root}/outputs/host-gpu-calibration.conf" ]]; then
  calibration="${repository_root}/outputs/host-gpu-calibration.conf"
fi

arguments=(
  run-dir
  --backend onnx
  --model "${model}"
  --input-dir "${input_dir}"
  --output-dir "${output_dir}"
  --report "${summary_path}"
  --input-width 512
  --input-height 512
  --classes 21
  --reserve-mb 1024
  --overlap-ratio "${overlap_ratio}"
  --warmup "${warmup}"
)
if [[ -n "${calibration}" ]]; then
  arguments+=(--calibration "${calibration}")
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
status=0
"${binary}" "${arguments[@]}" || status=$?

if [[ -f "${summary_path}" ]]; then
  jq '{
    total_images,
    successful_images,
    failed_images,
    warmup_runs,
    total_tiles,
    wall_time_ms,
    total_latency_ms,
    inference_latency_ms,
    mean_preprocess_ms,
    mean_merge_ms,
    execution_provider
  }' "${summary_path}"
  if python3 -c 'import PIL' >/dev/null 2>&1; then
    python3 "${repository_root}/scripts/analyze_batch.py" \
      --summary "${summary_path}" \
      --output-prefix "${output_dir}/analysis"
  fi
fi
exit "${status}"
