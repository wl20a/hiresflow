#!/usr/bin/env bash
set -euo pipefail

sample_dir="${1:-work/samples/nasa}"
sample_path="${sample_dir}/earth-day-2018.jpg"
sample_url="https://images-assets.nasa.gov/image/KSC-20180418-PH_FWM01_0013/KSC-20180418-PH_FWM01_0013~orig.JPG"
sample_sha="cc4829a344ee09c5471ba7e4be61645fd86277a5ae9d7aec01c1cb3a3f0185ac"

mkdir -p "${sample_dir}"
if [[ ! -f "${sample_path}" ]]; then
  wget -O "${sample_path}" "${sample_url}"
fi
printf '%s  %s\n' "${sample_sha}" "${sample_path}" | sha256sum -c -
echo "Sample is available at ${sample_path}"
