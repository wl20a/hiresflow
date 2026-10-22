#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-}"
if [[ -z "${image_path}" ]]; then
  echo "usage: $0 /path/to/large-image.(jpg|png)" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${HIRESFLOW_SWEEP_DIR:-${repository_root}/outputs/batch-sweep}"
batches="${HIRESFLOW_BATCHES:-1,2,4,8,16,32}"
warmup="${HIRESFLOW_SWEEP_WARMUP:-1}"
repeat="${HIRESFLOW_SWEEP_REPEAT:-2}"
mkdir -p "${output_dir}"

IFS=',' read -r -a batch_values <<< "${batches}"
for batch in "${batch_values[@]}"; do
  report="${output_dir}/batch-${batch}.json"
  echo "benchmark batch=${batch}"
  if HIRESFLOW_BATCH="${batch}" \
      HIRESFLOW_WARMUP="${warmup}" \
      HIRESFLOW_REPEAT="${repeat}" \
      HIRESFLOW_REPORT="${report}" \
      bash "${repository_root}/scripts/host_gpu_benchmark.sh" "${image_path}"; then
    jq -r --argjson batch "${batch}" '"batch=" + ($batch|tostring)
      + " p50_ms=" + (.total_latency_ms.p50|tostring)
      + " p95_ms=" + (.total_latency_ms.p95|tostring)
      + " inference_p50_ms=" + (.inference_latency_ms.p50|tostring)
      + " executed_batch=" + (.adaptive_batch.executed_batch_limit|tostring)
      + " peak_mib=" + ((.gpu_memory.peak_used_bytes / 1048576)|floor|tostring)' \
      "${report}"
  else
    echo "batch=${batch} failed" >&2
  fi
done
