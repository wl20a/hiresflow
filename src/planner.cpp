#include "hiresflow/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace hiresflow {
namespace {

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

int even_overlap(int tile_size, double ratio) {
  const int raw = std::max(0, static_cast<int>(std::lround(tile_size * ratio)));
  const int even = raw - raw % 2;
  return std::min(even, std::max(0, tile_size / 2));
}

int choose_tile_width(const PlannerRequest& request) {
  if (request.tile_size_candidates.empty()) {
    return request.model_input.width;
  }
  std::vector<int> sorted = request.tile_size_candidates;
  std::sort(sorted.begin(), sorted.end());
  for (const int candidate : sorted) {
    if (candidate >= request.model_input.width) {
      return candidate;
    }
  }
  return sorted.back();
}

std::uint64_t estimate_sample_bytes(const PlannerRequest& request) {
  if (request.calibrated_bytes_per_sample > 0) {
    return request.calibrated_bytes_per_sample;
  }

  const std::uint64_t pixels =
      static_cast<std::uint64_t>(request.model_input.width) *
      static_cast<std::uint64_t>(request.model_input.height);
  const std::uint64_t input_bytes = pixels * 3ULL * sizeof(float);
  const std::uint64_t logits_bytes =
      pixels * static_cast<std::uint64_t>(std::max(1, request.classes)) * sizeof(float);

  // The model's peak activation footprint is not derivable from its file size.
  // This deliberately conservative placeholder is replaced by calibration data
  // whenever the caller supplies a measured per-sample value.
  return input_bytes + logits_bytes * 2ULL + 64ULL * kMiB;
}

}  // namespace

ExecutionPlan create_plan(const PlannerRequest& request) {
  if (request.image.width <= 0 || request.image.height <= 0) {
    throw std::invalid_argument("planner requires a non-empty image");
  }
  if (request.model_input.width <= 0 || request.model_input.height <= 0) {
    throw std::invalid_argument("planner requires a positive model input size");
  }
  if (request.classes <= 0) {
    throw std::invalid_argument("planner requires at least one output class");
  }
  if (request.reserve_bytes >= request.gpu_budget_bytes) {
    throw std::invalid_argument("GPU reserve must be smaller than the total budget");
  }

  const int tile_width = choose_tile_width(request);
  const int tile_height = std::max(
      1,
      static_cast<int>(std::lround(
          static_cast<double>(tile_width) * request.model_input.height /
          request.model_input.width)));
  const int overlap_x = even_overlap(tile_width, request.overlap_ratio);
  const int overlap_y = even_overlap(tile_height, request.overlap_ratio);
  const std::uint64_t effective_budget =
      request.gpu_budget_bytes - request.reserve_bytes;
  const std::uint64_t sample_bytes = estimate_sample_bytes(request);
  const std::uint64_t fixed_bytes =
      request.model_bytes + 256ULL * kMiB;

  std::vector<int> batches = request.batch_candidates;
  if (batches.empty()) {
    batches = {1};
  }
  std::sort(batches.begin(), batches.end(), std::greater<int>());
  if (request.preferred_batch_size > 0 &&
      std::find(
          batches.begin(),
          batches.end(),
          request.preferred_batch_size) == batches.end()) {
    batches.push_back(request.preferred_batch_size);
    std::sort(batches.begin(), batches.end(), std::greater<int>());
  }

  int selected_batch = 0;
  std::uint64_t selected_peak = 0;
  const auto selection = [&](int batch) {
    if (batch <= 0 ||
        (request.max_batch_size > 0 && batch > request.max_batch_size)) {
      return false;
    }
    const std::uint64_t peak =
        fixed_bytes + static_cast<std::uint64_t>(batch) * sample_bytes;
    if (peak > effective_budget) {
      return false;
    }
    selected_batch = batch;
    selected_peak = peak;
    return true;
  };
  if (request.preferred_batch_size > 0 &&
      selection(request.preferred_batch_size)) {
    batches.clear();
  }
  for (const int batch : batches) {
    if (selection(batch)) {
      break;
    }
  }
  if (selected_batch == 0) {
    throw std::runtime_error(
        "no batch candidate fits both the model batch constraint and GPU budget; "
        "reduce model size, lower the batch, or increase the budget");
  }

  const auto tiles = make_tiles(
      request.image.width,
      request.image.height,
      tile_width,
      tile_height,
      overlap_x,
      overlap_y);

  ExecutionPlan plan;
  plan.image = request.image;
  plan.model_input = request.model_input;
  plan.tile_width = tile_width;
  plan.tile_height = tile_height;
  plan.overlap_x = overlap_x;
  plan.overlap_y = overlap_y;
  plan.batch_size = selected_batch;
  plan.streams = 1;
  plan.classes = request.classes;
  plan.precision = Precision::kFloat32;
  plan.estimated_peak_bytes = selected_peak;
  plan.effective_budget_bytes = effective_budget;
  plan.model_bytes = request.model_bytes;
  plan.estimated_sample_bytes = sample_bytes;
  plan.estimated_tile_count = static_cast<int>(tiles.size());
  plan.budget_source = request.budget_source;
  plan.planning_note =
      request.preferred_batch_size > 0 &&
              selected_batch == request.preferred_batch_size
          ? "batch selected from measured throughput calibration"
          : (request.calibrated_bytes_per_sample > 0
                 ? "batch selected from externally calibrated per-sample memory"
                 : "batch selected from conservative static memory estimate; calibrate on target GPU");
  return plan;
}

}  // namespace hiresflow
