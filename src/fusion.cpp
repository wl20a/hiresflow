#include "hiresflow/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hiresflow {
namespace {

constexpr int kInvalidLabel = -1;

}  // namespace

SegmentationFuser::SegmentationFuser(
    int width,
    int height,
    int classes,
    int candidates_per_pixel)
    : width_(width),
      height_(height),
      classes_(classes),
      entries_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
               static_cast<std::size_t>(std::max(1, candidates_per_pixel))) {
  if (width <= 0 || height <= 0 || classes <= 0 || candidates_per_pixel <= 0) {
    throw std::invalid_argument("invalid fusion buffer dimensions");
  }
}

void SegmentationFuser::add(
    const Tile& tile,
    const std::vector<std::uint16_t>& labels,
    const std::vector<float>& confidence,
    int label_width,
    int label_height) {
  if (tile.x < 0 || tile.y < 0 || tile.x + tile.width > width_ ||
      tile.y + tile.height > height_ || label_width <= 0 || label_height <= 0) {
    throw std::invalid_argument("tile lies outside the fusion buffer");
  }
  if (labels.size() !=
          static_cast<std::size_t>(label_width) * static_cast<std::size_t>(label_height) ||
      confidence.size() != labels.size()) {
    throw std::invalid_argument("tile labels and confidence buffers do not match");
  }

  const std::size_t candidates_per_pixel = entries_.size() /
      (static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_));
  const float scale = 1.0F / static_cast<float>(std::max(1, std::min(tile.width, tile.height)));
  for (int local_y = 0; local_y < tile.height; ++local_y) {
    const int label_y = std::min(
        label_height - 1,
        static_cast<int>(static_cast<long long>(local_y) * label_height / tile.height));
    const int global_y = tile.y + local_y;
    const int distance_y = std::min(local_y, tile.height - 1 - local_y);
    for (int local_x = 0; local_x < tile.width; ++local_x) {
      const int label_x = std::min(
          label_width - 1,
          static_cast<int>(static_cast<long long>(local_x) * label_width / tile.width));
      const int global_x = tile.x + local_x;
      const int distance_x = std::min(local_x, tile.width - 1 - local_x);
      const float spatial_weight =
          1.0F + 8.0F * static_cast<float>(std::min(distance_x, distance_y)) * scale;
      const std::size_t source_index =
          static_cast<std::size_t>(label_y) * static_cast<std::size_t>(label_width) +
          static_cast<std::size_t>(label_x);
      const std::uint16_t label16 = labels[source_index];
      if (label16 >= classes_) {
        throw std::runtime_error("backend emitted a label outside the configured class range");
      }
      const int label = static_cast<int>(label16);
      const float contribution =
          spatial_weight * std::max(0.0F, std::min(1.0F, confidence[source_index]));
      const std::size_t pixel =
          static_cast<std::size_t>(global_y) * static_cast<std::size_t>(width_) +
          static_cast<std::size_t>(global_x);
      const std::size_t base = pixel * candidates_per_pixel;

      std::size_t replacement = 0;
      bool merged = false;
      for (std::size_t candidate = 0; candidate < candidates_per_pixel; ++candidate) {
        Entry& entry = entries_[base + candidate];
        if (entry.label_index == label) {
          entry.weight += contribution;
          merged = true;
          break;
        }
        if (entry.label_index == kInvalidLabel) {
          entry.label_index = label;
          entry.weight = contribution;
          merged = true;
          break;
        }
        if (entry.weight < entries_[base + replacement].weight) {
          replacement = candidate;
        }
      }
      if (!merged && contribution > entries_[base + replacement].weight) {
        entries_[base + replacement] = Entry{label, contribution};
      }
    }
  }
}

MaskImage SegmentationFuser::finish() const {
  MaskImage mask;
  mask.width = width_;
  mask.height = height_;
  const std::size_t pixels =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  mask.labels.resize(pixels, 0);
  mask.confidence.resize(pixels, 0.0F);
  if (entries_.empty()) {
    return mask;
  }

  const std::size_t candidates_per_pixel = entries_.size() / pixels;
  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    const std::size_t base = pixel * candidates_per_pixel;
    const Entry* best = nullptr;
    for (std::size_t candidate = 0; candidate < candidates_per_pixel; ++candidate) {
      const Entry& entry = entries_[base + candidate];
      if (entry.label_index >= 0 &&
          (best == nullptr || entry.weight > best->weight)) {
        best = &entry;
      }
    }
    if (best != nullptr) {
      mask.labels[pixel] = static_cast<std::uint16_t>(best->label_index);
      mask.confidence[pixel] = std::max(0.0F, std::min(1.0F, best->weight / 9.0F));
    }
  }
  return mask;
}

}  // namespace hiresflow
