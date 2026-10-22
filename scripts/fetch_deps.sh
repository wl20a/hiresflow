#!/usr/bin/env bash
set -euo pipefail

deps_dir="${1:-work/deps}"
mkdir -p "${deps_dir}/stb"

onnxruntime_archive="${deps_dir}/onnxruntime.tgz"
onnxruntime_url="https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-gpu_cuda12-1.31.0.tgz"
onnxruntime_sha="62d3af5269af2143b5d37494337573f1ffb8885f3ad7477ae0cb1cc79c8dedfd"

if [[ "${HIRESFLOW_SKIP_ORT:-0}" != "1" ]]; then
  if [[ ! -f "${onnxruntime_archive}" ]]; then
    wget -c -O "${onnxruntime_archive}" "${onnxruntime_url}"
  fi
  printf '%s  %s\n' "${onnxruntime_sha}" "${onnxruntime_archive}" | sha256sum -c -
  tar -xzf "${onnxruntime_archive}" -C "${deps_dir}"
fi

fetch_with_sha() {
  local url="$1"
  local path="$2"
  local expected="$3"
  if [[ ! -f "${path}" ]]; then
    wget -q -O "${path}" "${url}"
  fi
  printf '%s  %s\n' "${expected}" "${path}" | sha256sum -c -
}

fetch_with_sha \
  "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" \
  "${deps_dir}/stb/stb_image.h" \
  "594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3"
fetch_with_sha \
  "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" \
  "${deps_dir}/stb/stb_image_write.h" \
  "cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05"
fetch_with_sha \
  "https://raw.githubusercontent.com/nothings/stb/master/LICENSE" \
  "${deps_dir}/stb/LICENSE" \
  "bebfe904b14301657e4e5d655c811d51fd31b97c455b9cc2d8600d6bac6cff63"

echo "Dependencies are available under ${deps_dir}"
