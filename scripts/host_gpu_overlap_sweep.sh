#!/usr/bin/env bash
set -euo pipefail

image_path="${1:-}"
if [[ -z "${image_path}" ]]; then
  echo "usage: $0 /path/to/large-image.(jpg|png)" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ratios="${HIRESFLOW_OVERLAPS:-0.125,0.25}"
output_dir="${HIRESFLOW_SWEEP_DIR:-${repository_root}/outputs/overlap-sweep}"
mkdir -p "${output_dir}"

IFS=',' read -r -a ratio_values <<< "${ratios}"
for ratio in "${ratio_values[@]}"; do
  pixels="$(awk -v ratio="${ratio}" 'BEGIN { printf "%d", ratio * 512 }')"
  prefix="${output_dir}/overlap-${pixels}"
  echo "overlap ratio=${ratio} (${pixels}px)"
  HIRESFLOW_OVERLAP="${ratio}" \
  HIRESFLOW_REPORT_PREFIX="${prefix}" \
    bash "${repository_root}/scripts/host_gpu_compare.sh" "${image_path}" || continue

  jq -r '
    "agreement=" + (.metrics.mask_agreement|tostring)
    + " seam_change=" + (.metrics.tile_seam_change_fraction|tostring)
    + " non_seam_change=" + (.metrics.non_seam_edge_change_fraction|tostring)
    + " tiled_ms=" + (.tiled.timing_ms.total|tostring)
    + " tiled_classes=" + (.label_coverage.tiled_present_labels|tostring)
    + " downscaled_classes=" + (.label_coverage.downscaled_present_labels|tostring)
  ' "${prefix}.report.json"
done
