#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hiresflow {

struct ImageSize {
  int width = 0;
  int height = 0;
};

struct Tile {
  int index = 0;
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;

  bool operator==(const Tile& other) const {
    return index == other.index && x == other.x && y == other.y &&
           width == other.width && height == other.height;
  }
};

enum class Precision {
  kFloat32,
  kFloat16,
};

inline const char* precision_name(Precision precision) {
  switch (precision) {
    case Precision::kFloat16:
      return "fp16";
    case Precision::kFloat32:
    default:
      return "fp32";
  }
}

struct PlannerRequest {
  ImageSize image;
  ImageSize model_input;
  int classes = 1;
  std::uint64_t gpu_budget_bytes = 6ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t reserve_bytes = 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t model_bytes = 0;
  std::uint64_t calibrated_bytes_per_sample = 0;
  std::string budget_source = "fallback";
  std::vector<int> tile_size_candidates{512};
  std::vector<int> batch_candidates{1, 2, 4, 8};
  int max_batch_size = 0;  // 0 means the model can accept a dynamic batch.
  int preferred_batch_size = 0;  // 0 means select by memory alone.
  double overlap_ratio = 0.125;
  bool allow_fp16 = false;
};

struct ExecutionPlan {
  ImageSize image;
  ImageSize model_input;
  int tile_width = 0;
  int tile_height = 0;
  int overlap_x = 0;
  int overlap_y = 0;
  int batch_size = 1;
  int streams = 1;
  int classes = 1;
  Precision precision = Precision::kFloat32;
  std::uint64_t estimated_peak_bytes = 0;
  std::uint64_t effective_budget_bytes = 0;
  std::uint64_t model_bytes = 0;
  std::uint64_t estimated_sample_bytes = 0;
  int estimated_tile_count = 0;
  std::string budget_source;
  std::string planning_note;
};

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;

  bool empty() const {
    return width <= 0 || height <= 0 || pixels.empty();
  }

  ImageSize size() const {
    return ImageSize{width, height};
  }
};

struct MaskImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint16_t> labels;
  std::vector<float> confidence;
};

struct TileTensor {
  Tile tile;
  int width = 0;
  int height = 0;
  int channels = 3;
  std::vector<float> data;  // CHW, normalized according to the backend.
};

struct ModelLogits {
  int classes = 0;
  int width = 0;
  int height = 0;
  std::vector<float> data;  // CHW
};

struct ModelSpec {
  std::string name;
  std::string execution_provider;
  ImageSize input;
  int classes = 1;
  int fixed_batch_size = 0;  // 0 means dynamic batch.
};

struct Normalization {
  float mean[3] = {0.485F, 0.456F, 0.406F};
  float stddev[3] = {0.229F, 0.224F, 0.225F};
  bool rgb_to_bgr = false;
};

struct StageTiming {
  double load_ms = 0.0;
  double plan_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double merge_ms = 0.0;
  double logits_decode_ms = 0.0;
  double output_resize_ms = 0.0;
  double fusion_ms = 0.0;
  double write_ms = 0.0;
  double total_ms = 0.0;
};

struct RunResult {
  MaskImage mask;
  ExecutionPlan plan;
  StageTiming timing;
  int tiles = 0;
  int batch_retries = 0;
  int executed_batch_limit = 0;
  bool gpu_memory_sampled = false;
  std::uint64_t gpu_baseline_used_bytes = 0;
  std::uint64_t gpu_peak_used_bytes = 0;
};

struct LatencyStats {
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct BenchmarkSummary {
  ExecutionPlan plan;
  int tiles = 0;
  int warmup_runs = 0;
  int measured_runs = 0;
  int batch_retries = 0;
  int executed_batch_limit = 0;
  LatencyStats total;
  LatencyStats inference;
  double mean_preprocess_ms = 0.0;
  double mean_merge_ms = 0.0;
  double mean_logits_decode_ms = 0.0;
  double mean_output_resize_ms = 0.0;
  double mean_fusion_ms = 0.0;
  bool gpu_memory_sampled = false;
  std::uint64_t gpu_peak_used_bytes = 0;
};

struct MaskAgreement {
  std::uint64_t pixels = 0;
  std::uint64_t equal_pixels = 0;
  double agreement = 0.0;
};

struct SeamDiagnostics {
  std::uint64_t seam_edges = 0;
  std::uint64_t seam_changes = 0;
  std::uint64_t non_seam_edges = 0;
  std::uint64_t non_seam_changes = 0;
  double seam_change_fraction = 0.0;
  double non_seam_change_fraction = 0.0;
};

struct MaskComparison {
  MaskAgreement agreement;
  SeamDiagnostics seams;
};

struct MaskLabelSummary {
  int present_labels = 0;
  std::uint64_t dominant_count = 0;
  double dominant_fraction = 0.0;
  std::vector<std::uint64_t> counts;
};

struct SegmentationEvaluation {
  int classes = 0;
  std::uint64_t pixels = 0;
  std::uint64_t correct_pixels = 0;
  double pixel_accuracy = 0.0;
  double mean_iou = 0.0;
  std::vector<double> class_iou;
  std::vector<std::uint64_t> intersections;
  std::vector<std::uint64_t> unions;
};

struct BatchImageSummary {
  std::string input_path;
  std::string output_path;
  bool success = false;
  std::string error;
  int tiles = 0;
  int batch_size = 0;
  int batch_retries = 0;
  double load_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double merge_ms = 0.0;
  double write_ms = 0.0;
  double total_ms = 0.0;
  std::uint64_t gpu_peak_used_bytes = 0;
};

struct BatchSummary {
  std::string backend_name;
  std::string execution_provider;
  int total_images = 0;
  int successful_images = 0;
  int failed_images = 0;
  int warmup_runs = 0;
  int total_tiles = 0;
  double wall_time_ms = 0.0;
  LatencyStats total_latency;
  LatencyStats inference_latency;
  double mean_preprocess_ms = 0.0;
  double mean_merge_ms = 0.0;
  std::vector<BatchImageSummary> images;
};

struct CalibrationEntry {
  int requested_batch = 0;
  int executed_batch = 0;
  bool success = false;
  std::string error;
  double total_p50_ms = 0.0;
  double total_p95_ms = 0.0;
  double inference_p50_ms = 0.0;
  std::uint64_t gpu_peak_used_bytes = 0;
};

struct CalibrationSummary {
  std::string execution_provider;
  std::string model_path;
  int selected_batch = 0;
  std::uint64_t calibrated_sample_bytes = 0;
  std::vector<CalibrationEntry> entries;
};

struct CalibrationConfig {
  int preferred_batch_size = 0;
  std::uint64_t calibrated_sample_bytes = 0;
  std::uint64_t model_bytes = 0;
  std::string execution_provider;
};

class SegmentationBackend {
 public:
  virtual ~SegmentationBackend() = default;
  virtual const ModelSpec& spec() const = 0;
  virtual std::vector<ModelLogits> infer(const std::vector<TileTensor>& batch) = 0;
};

}  // namespace hiresflow
