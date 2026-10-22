#include "hiresflow/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace hiresflow {
namespace {

std::vector<int> tile_positions(int image_size, int tile_size, int overlap) {
  if (image_size <= 0 || tile_size <= 0) {
    return {};
  }
  if (tile_size >= image_size) {
    return {0};
  }

  const int stride = tile_size - overlap;
  if (stride <= 0) {
    throw std::invalid_argument("tile overlap must be smaller than tile size");
  }
  const int span = image_size - tile_size;
  const int intervals =
      std::max(1, static_cast<int>(std::ceil(
                      static_cast<double>(span) / static_cast<double>(stride))));
  std::vector<int> positions;
  positions.reserve(static_cast<std::size_t>(intervals + 1));
  for (int index = 0; index <= intervals; ++index) {
    const int position = static_cast<int>(
        static_cast<long long>(index) * span / intervals);
    if (positions.empty() || positions.back() != position) {
      positions.push_back(position);
    }
  }
  return positions;
}

}  // namespace

std::vector<Tile> make_tiles(
    int image_width,
    int image_height,
    int tile_width,
    int tile_height,
    int overlap_x,
    int overlap_y) {
  if (image_width <= 0 || image_height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  if (tile_width <= 0 || tile_height <= 0) {
    throw std::invalid_argument("tile dimensions must be positive");
  }
  if (overlap_x < 0 || overlap_y < 0) {
    throw std::invalid_argument("tile overlap cannot be negative");
  }

  const int effective_width = std::min(tile_width, image_width);
  const int effective_height = std::min(tile_height, image_height);
  const int effective_overlap_x = std::min(overlap_x, effective_width - 1);
  const int effective_overlap_y = std::min(overlap_y, effective_height - 1);
  const std::vector<int> xs =
      tile_positions(image_width, effective_width, effective_overlap_x);
  const std::vector<int> ys =
      tile_positions(image_height, effective_height, effective_overlap_y);

  std::vector<Tile> tiles;
  tiles.reserve(xs.size() * ys.size());
  int index = 0;
  for (const int y : ys) {
    for (const int x : xs) {
      tiles.push_back(Tile{index++, x, y, effective_width, effective_height});
    }
  }
  return tiles;
}

}  // namespace hiresflow
