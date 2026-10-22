#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-work/models}"
model_path="${model_dir}/fcn-resnet50-12.onnx"
model_url="https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx"
model_sha="eb5017d1b80372eb0b58552655274698817ba2e774437e2c2d3c0a613d2e99bd"

mkdir -p "${model_dir}"
if [[ ! -f "${model_path}" ]]; then
  wget -c -O "${model_path}" "${model_url}"
fi
printf '%s  %s\n' "${model_sha}" "${model_path}" | sha256sum -c -

echo "Model is available at ${model_path}"
