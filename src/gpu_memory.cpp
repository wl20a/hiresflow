#include "hiresflow/gpu_memory.hpp"

#include <cuda_runtime_api.h>

namespace hiresflow {

std::optional<GpuMemorySample> sample_gpu_memory() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return std::nullopt;
  }
  return GpuMemorySample{
      static_cast<std::uint64_t>(total_bytes),
      static_cast<std::uint64_t>(free_bytes)};
}

}  // namespace hiresflow
