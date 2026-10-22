#include "hiresflow/core.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace hiresflow {

MaskAgreement compare_masks(const MaskImage& reference, const MaskImage& candidate) {
  if (reference.width != candidate.width || reference.height != candidate.height ||
      reference.labels.size() != candidate.labels.size()) {
    throw std::invalid_argument("cannot compare masks with different dimensions");
  }
  MaskAgreement agreement;
  agreement.pixels = reference.labels.size();
  for (std::size_t index = 0; index < reference.labels.size(); ++index) {
    if (reference.labels[index] == candidate.labels[index]) {
      ++agreement.equal_pixels;
    }
  }
  agreement.agreement = agreement.pixels == 0
      ? 0.0
      : static_cast<double>(agreement.equal_pixels) /
            static_cast<double>(agreement.pixels);
  return agreement;
}

MaskLabelSummary summarize_mask_labels(const MaskImage& mask, int classes) {
  if (classes <= 0 || mask.labels.empty()) {
    throw std::invalid_argument("label summary requires labels and a positive class count");
  }
  MaskLabelSummary summary;
  summary.counts.assign(static_cast<std::size_t>(classes), 0);
  for (const std::uint16_t label : mask.labels) {
    if (label >= classes) {
      throw std::runtime_error("mask contains a class outside the summary range");
    }
    ++summary.counts[static_cast<std::size_t>(label)];
  }
  for (const std::uint64_t count : summary.counts) {
    if (count > 0) {
      ++summary.present_labels;
    }
    summary.dominant_count = std::max(summary.dominant_count, count);
  }
  summary.dominant_fraction =
      static_cast<double>(summary.dominant_count) /
      static_cast<double>(mask.labels.size());
  return summary;
}

SeamDiagnostics diagnose_tile_seams(
    const MaskImage& mask,
    const ExecutionPlan& tiled_plan) {
  if (mask.width <= 0 || mask.height <= 0 ||
      mask.labels.size() !=
          static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height)) {
    throw std::invalid_argument("seam diagnostics require a valid mask");
  }
  const std::vector<Tile> tiles = make_tiles(
      mask.width,
      mask.height,
      tiled_plan.tile_width,
      tiled_plan.tile_height,
      tiled_plan.overlap_x,
      tiled_plan.overlap_y);
  std::vector<bool> vertical_seam(static_cast<std::size_t>(mask.width), false);
  std::vector<bool> horizontal_seam(static_cast<std::size_t>(mask.height), false);
  for (const Tile& tile : tiles) {
    if (tile.x > 0 && tile.x < mask.width) {
      vertical_seam[static_cast<std::size_t>(tile.x)] = true;
    }
    const int right = tile.x + tile.width;
    if (right > 0 && right < mask.width) {
      vertical_seam[static_cast<std::size_t>(right)] = true;
    }
    if (tile.y > 0 && tile.y < mask.height) {
      horizontal_seam[static_cast<std::size_t>(tile.y)] = true;
    }
    const int bottom = tile.y + tile.height;
    if (bottom > 0 && bottom < mask.height) {
      horizontal_seam[static_cast<std::size_t>(bottom)] = true;
    }
  }

  SeamDiagnostics diagnostics;
  const auto account = [&diagnostics](bool seam, bool changed) {
    if (seam) {
      ++diagnostics.seam_edges;
      diagnostics.seam_changes += changed ? 1U : 0U;
    } else {
      ++diagnostics.non_seam_edges;
      diagnostics.non_seam_changes += changed ? 1U : 0U;
    }
  };

  for (int y = 0; y < mask.height; ++y) {
    for (int x = 1; x < mask.width; ++x) {
      const std::size_t left =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(mask.width) +
          static_cast<std::size_t>(x - 1);
      const std::size_t right = left + 1U;
      account(
          vertical_seam[static_cast<std::size_t>(x)],
          mask.labels[left] != mask.labels[right]);
    }
  }
  for (int y = 1; y < mask.height; ++y) {
    const bool seam = horizontal_seam[static_cast<std::size_t>(y)];
    for (int x = 0; x < mask.width; ++x) {
      const std::size_t top =
          static_cast<std::size_t>(y - 1) * static_cast<std::size_t>(mask.width) +
          static_cast<std::size_t>(x);
      const std::size_t bottom =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(mask.width) +
          static_cast<std::size_t>(x);
      account(seam, mask.labels[top] != mask.labels[bottom]);
    }
  }

  diagnostics.seam_change_fraction = diagnostics.seam_edges == 0
      ? 0.0
      : static_cast<double>(diagnostics.seam_changes) /
            static_cast<double>(diagnostics.seam_edges);
  diagnostics.non_seam_change_fraction = diagnostics.non_seam_edges == 0
      ? 0.0
      : static_cast<double>(diagnostics.non_seam_changes) /
            static_cast<double>(diagnostics.non_seam_edges);
  return diagnostics;
}

}  // namespace hiresflow
