#pragma once

#include <cstdint>
#include <optional>

namespace hiresflow {

struct GpuMemorySample {
  std::uint64_t total_bytes = 0;
  std::uint64_t free_bytes = 0;
};

std::optional<GpuMemorySample> sample_gpu_memory();

}  // namespace hiresflow
