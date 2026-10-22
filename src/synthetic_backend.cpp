#include "hiresflow/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hiresflow {

SyntheticSegmentationBackend::SyntheticSegmentationBackend(int input_size) {
  if (input_size <= 0) {
    throw std::invalid_argument("synthetic input size must be positive");
  }
  spec_.name = "synthetic-segmentation";
  spec_.execution_provider = "cpu";
  spec_.input = ImageSize{input_size, input_size};
  spec_.classes = 8;
}

const ModelSpec& SyntheticSegmentationBackend::spec() const {
  return spec_;
}

std::vector<ModelLogits> SyntheticSegmentationBackend::infer(
    const std::vector<TileTensor>& batch) {
  outputs_.clear();
  outputs_.reserve(batch.size());
  for (const TileTensor& tile : batch) {
    if (tile.width != spec_.input.width || tile.height != spec_.input.height) {
      throw std::runtime_error("synthetic backend received an unexpected tensor shape");
    }
    ModelLogits output;
    output.classes = spec_.classes;
    output.width = tile.width;
    output.height = tile.height;
    output.data.assign(
        static_cast<std::size_t>(output.classes) * static_cast<std::size_t>(output.width) *
            static_cast<std::size_t>(output.height),
        -4.0F);
    const std::size_t plane =
        static_cast<std::size_t>(output.width) * static_cast<std::size_t>(output.height);
    for (int y = 0; y < output.height; ++y) {
      for (int x = 0; x < output.width; ++x) {
        const int global_x = tile.tile.x + x;
        const int global_y = tile.tile.y + y;
        const int label = ((global_x / 96) + (global_y / 96)) % output.classes;
        output.data[static_cast<std::size_t>(label) * plane +
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(output.width) +
                    static_cast<std::size_t>(x)] = 4.0F;
      }
    }
    outputs_.push_back(std::move(output));
  }
  return outputs_;
}

}  // namespace hiresflow
