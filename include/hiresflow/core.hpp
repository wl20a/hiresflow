#pragma once

#include "hiresflow/types.hpp"

#include <string>
#include <vector>

namespace hiresflow {

bool load_rgb_image(const std::string& path, RgbImage& image, std::string& error);
bool write_rgb_png(const std::string& path, const RgbImage& image, std::string& error);
bool write_mask_png(const std::string& path, const MaskImage& mask, std::string& error);
bool load_index_mask_png(const std::string& path, MaskImage& mask, std::string& error);
bool write_index_mask_png(const std::string& path, const MaskImage& mask, std::string& error);

RgbImage crop_rgb(const RgbImage& image, const Tile& tile);
RgbImage resize_rgb_bilinear(const RgbImage& image, int width, int height);
std::vector<std::uint16_t> resize_mask_nearest(
    const std::vector<std::uint16_t>& labels,
    int source_width,
    int source_height,
    int target_width,
    int target_height);
std::vector<float> resize_scalar_bilinear(
    const std::vector<float>& values,
    int source_width,
    int source_height,
    int target_width,
    int target_height);

std::vector<Tile> make_tiles(
    int image_width,
    int image_height,
    int tile_width,
    int tile_height,
    int overlap_x,
    int overlap_y);

ExecutionPlan create_plan(const PlannerRequest& request);

class SegmentationFuser {
 public:
  SegmentationFuser(int width, int height, int classes, int candidates_per_pixel = 4);

  void add(const Tile& tile,
           const std::vector<std::uint16_t>& labels,
           const std::vector<float>& confidence,
           int label_width,
           int label_height);

  MaskImage finish() const;

 private:
  struct Entry {
    int label_index = -1;
    float weight = 0.0F;
  };

  int width_;
  int height_;
  int classes_;
  std::vector<Entry> entries_;
};

class SyntheticSegmentationBackend final : public SegmentationBackend {
 public:
  explicit SyntheticSegmentationBackend(int input_size = 512);
  const ModelSpec& spec() const override;
  std::vector<ModelLogits> infer(const std::vector<TileTensor>& batch) override;

 private:
  ModelSpec spec_;
  std::vector<ModelLogits> outputs_;
};

RunResult run_segmentation(
    const RgbImage& image,
    const ExecutionPlan& plan,
    SegmentationBackend& backend,
    const Normalization& normalization = {});

RunResult run_downscaled_segmentation(
    const RgbImage& image,
    SegmentationBackend& backend,
    const Normalization& normalization = {});

MaskAgreement compare_masks(const MaskImage& reference, const MaskImage& candidate);
MaskLabelSummary summarize_mask_labels(const MaskImage& mask, int classes);
SeamDiagnostics diagnose_tile_seams(
    const MaskImage& mask,
    const ExecutionPlan& tiled_plan);
SegmentationEvaluation evaluate_segmentation(
    const MaskImage& prediction,
    const MaskImage& ground_truth,
    int classes);

bool write_report_json(
    const std::string& path,
    const RunResult& result,
    const std::string& input_path,
    const std::string& output_path,
    const std::string& backend_name,
    const std::string& execution_provider,
    const std::string& model_path,
    std::string& error);

bool write_plan_json(
    const std::string& path,
    const ExecutionPlan& plan,
    std::string& error);

bool write_benchmark_json(
    const std::string& path,
    const BenchmarkSummary& summary,
    const std::string& input_path,
    const std::string& model_path,
    const std::string& backend_name,
    const std::string& execution_provider,
    std::string& error);

bool write_comparison_json(
    const std::string& path,
    const MaskComparison& comparison,
    const RunResult& tiled,
    const RunResult& downscaled,
    const std::string& input_path,
    const std::string& tiled_output_path,
    const std::string& downscaled_output_path,
    const std::string& model_path,
    const std::string& backend_name,
    const std::string& execution_provider,
    const SegmentationEvaluation* tiled_evaluation,
    const SegmentationEvaluation* downscaled_evaluation,
    std::string& error);

bool write_evaluation_json(
    const std::string& path,
    const SegmentationEvaluation& evaluation,
    const std::string& prediction_path,
    const std::string& ground_truth_path,
    std::string& error);

bool write_batch_summary_json(
    const std::string& path,
    const BatchSummary& summary,
    std::string& error);

bool write_calibration_json(
    const std::string& path,
    const CalibrationSummary& summary,
    std::string& error);

bool write_calibration_config(
    const std::string& path,
    const CalibrationSummary& summary,
    std::uint64_t model_bytes,
    std::string& error);

bool load_calibration_config(
    const std::string& path,
    CalibrationConfig& config,
    std::string& error);

}  // namespace hiresflow

#ifdef HIRESFLOW_HAS_ORT
#include "hiresflow/onnx_backend.hpp"
#endif
