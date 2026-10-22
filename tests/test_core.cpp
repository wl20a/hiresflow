#include "hiresflow/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_tile_coverage() {
  constexpr int width = 1037;
  constexpr int height = 829;
  const std::vector<hiresflow::Tile> tiles =
      hiresflow::make_tiles(width, height, 256, 192, 32, 24);
  std::vector<int> coverage(static_cast<std::size_t>(width) * height, 0);
  for (const hiresflow::Tile& tile : tiles) {
    require(tile.x >= 0 && tile.y >= 0, "tile position is negative");
    require(tile.x + tile.width <= width, "tile exceeds image width");
    require(tile.y + tile.height <= height, "tile exceeds image height");
    for (int y = tile.y; y < tile.y + tile.height; ++y) {
      for (int x = tile.x; x < tile.x + tile.width; ++x) {
        ++coverage[static_cast<std::size_t>(y) * width + x];
      }
    }
  }
  require(
      std::all_of(coverage.begin(), coverage.end(), [](int count) { return count > 0; }),
      "tiles do not cover the complete image");
  require(tiles.back().x + tiles.back().width == width, "last column is not edge-anchored");
  require(tiles.back().y + tiles.back().height == height, "last row is not edge-anchored");
}

void test_tile_positions_are_evenly_distributed() {
  constexpr int width = 1000;
  constexpr int height = 900;
  constexpr int tile = 512;
  constexpr int overlap = 64;
  const std::vector<hiresflow::Tile> tiles =
      hiresflow::make_tiles(width, height, tile, tile, overlap, overlap);
  const std::vector<int> expected_x{0, 244, 488};
  const std::vector<int> expected_y{0, 388};
  std::vector<int> xs;
  std::vector<int> ys;
  for (const hiresflow::Tile& value : tiles) {
    if (std::find(xs.begin(), xs.end(), value.x) == xs.end()) {
      xs.push_back(value.x);
    }
    if (std::find(ys.begin(), ys.end(), value.y) == ys.end()) {
      ys.push_back(value.y);
    }
  }
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  require(xs == expected_x, "horizontal tile positions are not evenly distributed");
  require(ys == expected_y, "vertical tile positions are not evenly distributed");
}

void test_planner_respects_budget() {
  hiresflow::PlannerRequest request;
  request.image = hiresflow::ImageSize{2048, 1536};
  request.model_input = hiresflow::ImageSize{128, 128};
  request.classes = 8;
  request.gpu_budget_bytes = 512ULL * 1024ULL * 1024ULL;
  request.reserve_bytes = 128ULL * 1024ULL * 1024ULL;
  request.model_bytes = 32ULL * 1024ULL * 1024ULL;
  request.tile_size_candidates = {128};
  request.batch_candidates = {1, 2, 4, 8};
  request.calibrated_bytes_per_sample = 32ULL * 1024ULL * 1024ULL;

  const hiresflow::ExecutionPlan plan = hiresflow::create_plan(request);
  require(plan.batch_size > 0, "planner selected no batch");
  require(
      plan.estimated_peak_bytes <= plan.effective_budget_bytes,
      "planner exceeded the effective memory budget");
  require(plan.estimated_tile_count > 0, "planner created no tiles");
}

void test_planner_respects_fixed_batch() {
  hiresflow::PlannerRequest request;
  request.image = hiresflow::ImageSize{640, 480};
  request.model_input = hiresflow::ImageSize{128, 128};
  request.classes = 4;
  request.gpu_budget_bytes = 1024ULL * 1024ULL * 1024ULL;
  request.reserve_bytes = 128ULL * 1024ULL * 1024ULL;
  request.calibrated_bytes_per_sample = 1ULL * 1024ULL * 1024ULL;
  request.batch_candidates = {1, 2, 4, 8};
  request.max_batch_size = 1;
  const hiresflow::ExecutionPlan plan = hiresflow::create_plan(request);
  require(plan.batch_size == 1, "planner ignored the fixed model batch size");
}

void test_fusion_prefers_confident_tile() {
  hiresflow::SegmentationFuser fuser(6, 4, 3);
  const std::vector<hiresflow::Tile> tiles =
      hiresflow::make_tiles(6, 4, 4, 4, 2, 0);
  require(tiles.size() == 2, "expected two overlapping tiles");

  const std::vector<std::uint16_t> labels1(16, 1);
  const std::vector<std::uint16_t> labels2(16, 2);
  const std::vector<float> confidence1(16, 0.4F);
  const std::vector<float> confidence2(16, 0.9F);
  fuser.add(tiles[0], labels1, confidence1, 4, 4);
  fuser.add(tiles[1], labels2, confidence2, 4, 4);
  const hiresflow::MaskImage mask = fuser.finish();

  require(mask.labels[1] == 1, "non-overlapping tile label was changed");
  require(mask.labels[4] == 2, "more confident overlapping tile did not win");
}

void test_image_round_trip() {
  hiresflow::RgbImage image;
  image.width = 5;
  image.height = 3;
  image.pixels.resize(5 * 3 * 3);
  for (std::size_t index = 0; index < image.pixels.size(); ++index) {
    image.pixels[index] = static_cast<std::uint8_t>((index * 17U) % 255U);
  }

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "hiresflow_round_trip.png";
  std::string error;
  require(hiresflow::write_rgb_png(path.string(), image, error), error);
  hiresflow::RgbImage loaded;
  require(hiresflow::load_rgb_image(path.string(), loaded, error), error);
  std::filesystem::remove(path);
  require(loaded.width == image.width && loaded.height == image.height, "image size changed");
  require(loaded.pixels == image.pixels, "PNG round trip changed pixels");
}

void test_synthetic_pipeline() {
  hiresflow::RgbImage image;
  image.width = 37;
  image.height = 23;
  image.pixels.assign(37U * 23U * 3U, 127U);

  hiresflow::SyntheticSegmentationBackend backend(16);
  hiresflow::PlannerRequest request;
  request.image = image.size();
  request.model_input = backend.spec().input;
  request.classes = backend.spec().classes;
  request.tile_size_candidates = {16};
  request.batch_candidates = {1, 2, 4};
  request.overlap_ratio = 0.25;
  request.gpu_budget_bytes = 512ULL * 1024ULL * 1024ULL;
  request.reserve_bytes = 64ULL * 1024ULL * 1024ULL;
  const hiresflow::ExecutionPlan plan = hiresflow::create_plan(request);
  const hiresflow::RunResult result =
      hiresflow::run_segmentation(image, plan, backend);
  const hiresflow::RunResult downscaled =
      hiresflow::run_downscaled_segmentation(image, backend);

  require(result.mask.width == image.width, "output width changed");
  require(result.mask.height == image.height, "output height changed");
  require(
      downscaled.mask.width == image.width &&
          downscaled.mask.height == image.height,
      "downscaled baseline changed the output size");
  require(
      *std::max_element(result.mask.labels.begin(), result.mask.labels.end()) <
          backend.spec().classes,
      "synthetic pipeline emitted an invalid class");
  const hiresflow::SeamDiagnostics seams =
      hiresflow::diagnose_tile_seams(result.mask, plan);
  require(seams.seam_edges > 0, "seam diagnostics did not find tile boundaries");
  require(
      seams.seam_change_fraction >= 0.0 && seams.seam_change_fraction <= 1.0,
      "seam change fraction is outside [0,1]");
}

void test_mask_agreement() {
  hiresflow::MaskImage first;
  first.width = 3;
  first.height = 2;
  first.labels = {0, 1, 2, 2, 1, 0};
  hiresflow::MaskImage second = first;
  second.labels[1] = 2;
  second.labels[4] = 2;
  const hiresflow::MaskAgreement agreement =
      hiresflow::compare_masks(first, second);
  require(agreement.pixels == 6, "mask agreement pixel count is wrong");
  require(agreement.equal_pixels == 4, "mask agreement equal count is wrong");
  require(
      std::abs(agreement.agreement - 4.0 / 6.0) < 1.0e-9,
      "mask agreement fraction is wrong");
  const hiresflow::MaskLabelSummary summary =
      hiresflow::summarize_mask_labels(first, 3);
  require(summary.present_labels == 3, "label summary class count is wrong");
}

void test_index_mask_and_metrics() {
  hiresflow::MaskImage prediction;
  prediction.width = 2;
  prediction.height = 2;
  prediction.labels = {0, 1, 1, 0};
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "hiresflow_index_mask.png";
  std::string error;
  require(hiresflow::write_index_mask_png(path.string(), prediction, error), error);
  hiresflow::MaskImage loaded;
  require(hiresflow::load_index_mask_png(path.string(), loaded, error), error);
  std::filesystem::remove(path);
  require(loaded.labels == prediction.labels, "index PNG round trip changed labels");

  hiresflow::MaskImage truth = prediction;
  truth.labels[0] = 1;
  const hiresflow::SegmentationEvaluation evaluation =
      hiresflow::evaluate_segmentation(prediction, truth, 2);
  require(evaluation.correct_pixels == 3, "evaluation correct-pixel count is wrong");
  require(
      std::abs(evaluation.pixel_accuracy - 0.75) < 1.0e-9,
      "pixel accuracy is wrong");
}

class OomOnceBackend final : public hiresflow::SegmentationBackend {
 public:
  OomOnceBackend() {
    spec_.name = "oom-test";
    spec_.execution_provider = "cpu";
    spec_.input = hiresflow::ImageSize{8, 8};
    spec_.classes = 2;
  }

  const hiresflow::ModelSpec& spec() const override {
    return spec_;
  }

  std::vector<hiresflow::ModelLogits> infer(
      const std::vector<hiresflow::TileTensor>& batch) override {
    if (batch.size() > 1 && !failed_) {
      failed_ = true;
      throw std::runtime_error("Failed to allocate memory for requested buffer");
    }
    std::vector<hiresflow::ModelLogits> outputs;
    outputs.reserve(batch.size());
    for (const auto& tile : batch) {
      hiresflow::ModelLogits output;
      output.classes = 2;
      output.width = tile.width;
      output.height = tile.height;
      output.data.assign(
          static_cast<std::size_t>(2 * output.width * output.height),
          -1.0F);
      const std::size_t plane =
          static_cast<std::size_t>(output.width) * output.height;
      for (std::size_t pixel = 0; pixel < plane; ++pixel) {
        output.data[pixel] = 1.0F;
      }
      outputs.push_back(std::move(output));
    }
    return outputs;
  }

 private:
  hiresflow::ModelSpec spec_;
  bool failed_ = false;
};

void test_oom_backoff() {
  hiresflow::RgbImage image;
  image.width = 24;
  image.height = 8;
  image.pixels.assign(24U * 8U * 3U, 64U);
  OomOnceBackend backend;
  hiresflow::ExecutionPlan plan;
  plan.image = image.size();
  plan.model_input = backend.spec().input;
  plan.tile_width = 8;
  plan.tile_height = 8;
  plan.overlap_x = 0;
  plan.overlap_y = 0;
  plan.batch_size = 4;
  plan.streams = 1;
  plan.classes = 2;
  plan.estimated_tile_count = 3;
  const hiresflow::RunResult result =
      hiresflow::run_segmentation(image, plan, backend);
  require(result.batch_retries > 0, "OOM did not trigger a batch retry");
  require(result.executed_batch_limit < 4, "OOM retry did not lower the batch limit");
}

void test_calibration_config_round_trip() {
  hiresflow::CalibrationSummary summary;
  summary.execution_provider = "CUDAExecutionProvider";
  summary.selected_batch = 2;
  summary.calibrated_sample_bytes = 123456;
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "hiresflow_calibration.conf";
  std::string error;
  require(
      hiresflow::write_calibration_config(
          path.string(), summary, 987654, error),
      error);
  hiresflow::CalibrationConfig config;
  require(hiresflow::load_calibration_config(path.string(), config, error), error);
  std::filesystem::remove(path);
  require(config.preferred_batch_size == 2, "calibration batch did not round trip");
  require(
      config.calibrated_sample_bytes == 123456,
      "calibration memory did not round trip");
  require(config.model_bytes == 987654, "calibration model size did not round trip");
  require(
      config.execution_provider == "CUDAExecutionProvider",
      "calibration provider did not round trip");
}

}  // namespace

int main() {
  try {
    test_tile_coverage();
    test_tile_positions_are_evenly_distributed();
    test_planner_respects_budget();
    test_planner_respects_fixed_batch();
    test_fusion_prefers_confident_tile();
    test_image_round_trip();
    test_synthetic_pipeline();
    test_mask_agreement();
    test_index_mask_and_metrics();
    test_oom_backoff();
    test_calibration_config_round_trip();
    std::cout << "all tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "test failure: " << exception.what() << "\n";
    return 1;
  }
}
