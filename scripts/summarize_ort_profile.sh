#!/usr/bin/env bash
set -euo pipefail

profile_path="${1:-}"
if [[ -z "${profile_path}" || ! -f "${profile_path}" ]]; then
  echo "usage: $0 /path/to/hiresflow_ort_profile_*.json" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to summarize an ONNX Runtime profile" >&2
  exit 2
fi

echo "Provider node counts and aggregate kernel time:"
jq -r '.[] | select(.cat == "Node") | [.args.provider, .dur] | @tsv' "${profile_path}" |
  awk -F '\t' '{count[$1]++; duration[$1] += $2}
    END {for (provider in count)
      printf "  %s: %d nodes, %.1f ms\n", provider, count[provider], duration[provider] / 1000}' |
  sort

echo "Top operators by aggregate kernel time:"
jq -r '.[] | select(.cat == "Node") | [.args.op_name, .dur] | @tsv' "${profile_path}" |
  awk -F '\t' '{count[$1]++; duration[$1] += $2}
    END {for (op in count)
      printf "  %-24s %4d nodes  %10.1f ms\n", op, count[op], duration[op] / 1000}' |
  sort -k4,4nr |
  sed -n '1,20p'
