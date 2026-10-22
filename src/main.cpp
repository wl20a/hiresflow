#include "hiresflow/core.hpp"

#ifdef HIRESFLOW_HAS_CUDA
#include "hiresflow/gpu_memory.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hiresflow::ExecutionPlan;
using hiresflow::ImageSize;
using hiresflow::MaskImage;
using hiresflow::PlannerRequest;
using hiresflow::RgbImage;
using hiresflow::RunResult;

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

struct Options {
  std::string command;
  std::map<std::string, std::string> values;

  bool has(const std::string& key) const {
    return values.find(key) != values.end();
  }

  std::string get(const std::string& key, const std::string& fallback = "") const {
    const auto found = values.find(key);
    return found == values.end() ? fallback : found->second;
  }

  int get_int(const std::string& key, int fallback) const {
    const std::string value = get(key);
    return value.empty() ? fallback : std::stoi(value);
  }

  double get_double(const std::string& key, double fallback) const {
    const std::string value = get(key);
    return value.empty() ? fallback : std::stod(value);
  }
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument("missing command");
  }
  Options options;
  options.command = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument.rfind("--", 0) != 0) {
      throw std::invalid_argument("unexpected positional argument '" + argument + "'");
    }
    if (argument == "--cpu") {
      options.values["cpu"] = "true";
      continue;
    }
    if (argument == "--profile") {
      options.values["profile"] = "true";
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for option '" + argument + "'");
    }
    options.values[argument.substr(2)] = argv[++index];
  }
  return options;
}

void print_usage() {
  std::cout
      << "HiResFlow 0.1.0-preview\n\n"
      << "Usage:\n"
      << "  hiresflow plan --input IMAGE [options]\n"
      << "  hiresflow run  --input IMAGE --output mask.png [options]\n\n"
      << "  hiresflow benchmark --input IMAGE [options]\n\n"
      << "  hiresflow compare --input IMAGE --output PREFIX [options]\n\n"
      << "  hiresflow evaluate --prediction INDEX.png --ground-truth INDEX.png [options]\n\n"
      << "  hiresflow run-dir --input-dir DIR --output-dir DIR [options]\n\n"
      << "  hiresflow calibrate --input IMAGE --model PATH [options]\n\n"
      << "Run options:\n"
      << "  --backend onnx|synthetic       Backend (default: onnx)\n"
      << "  --model PATH                  ONNX model path\n"
      << "  --cpu                         Force ONNX Runtime CPU provider\n"
      << "  --profile                     Write an ONNX Runtime profile JSON\n"
      << "  --budget-mb N                 Total GPU budget (default: 6144)\n"
      << "  --reserve-mb N                Safety reserve (default: 1024)\n"
      << "  --batch N                     Force one batch candidate\n"
      << "  --preferred-batch N           Prefer a measured batch size\n"
      << "  --calibration PATH            Load measured batch calibration\n"
      << "  --tile-size LIST              Comma-separated tile sizes\n"
      << "  --overlap-ratio R             Source-tile overlap ratio (default: 0.125)\n"
      << "  --calibrated-sample-mb N      Measured per-sample working memory\n"
      << "  --classes N                   Override segmentation class count\n"
      << "  --ground-truth PATH           Single-channel index mask for mIoU\n"
      << "  --normalization MODE          imagenet or none (default: imagenet)\n"
      << "  --bgr                         Convert RGB input channels to BGR\n"
      << "  --input-width N               Override model input width\n"
      << "  --input-height N              Override model input height\n"
      << "  --index-output PATH           Also write an 8-bit class-index PNG\n"
      << "  --report PATH                 JSON report path\n";
}

std::vector<int> parse_int_list(const std::string& value) {
  std::vector<int> values;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::string token =
        value.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
    if (!token.empty()) {
      values.push_back(std::stoi(token));
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  return values;
}

std::uint64_t file_size(const std::string& path) {
  if (path.empty()) {
    return 0;
  }
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  return error ? 0 : static_cast<std::uint64_t>(size);
}

PlannerRequest make_planner_request(
    const Options& options,
    const RgbImage& image,
    const hiresflow::ModelSpec& spec) {
  PlannerRequest request;
  request.image = image.size();
  request.model_input = spec.input;
  request.classes = spec.classes;
  request.max_batch_size = spec.fixed_batch_size;
  if (options.has("budget-mb")) {
    request.gpu_budget_bytes =
        static_cast<std::uint64_t>(options.get_int("budget-mb", 6144)) * kMiB;
    request.budget_source = "explicit --budget-mb";
  } else {
    request.gpu_budget_bytes = 6144ULL * kMiB;
    request.budget_source = "fallback default 6144 MiB";
#ifdef HIRESFLOW_HAS_CUDA
    if (spec.execution_provider == "CUDAExecutionProvider") {
      if (const auto sample = hiresflow::sample_gpu_memory()) {
        request.gpu_budget_bytes = sample->free_bytes;
        request.budget_source = "automatic free CUDA device memory";
      }
    }
#endif
  }
  request.reserve_bytes =
      static_cast<std::uint64_t>(options.get_int("reserve-mb", 1024)) * kMiB;
  request.model_bytes = file_size(options.get("model"));
  request.overlap_ratio = options.get_double("overlap-ratio", 0.125);
  if (options.has("calibration")) {
    hiresflow::CalibrationConfig calibration;
    std::string calibration_error;
    if (!hiresflow::load_calibration_config(
            options.get("calibration"), calibration, calibration_error)) {
      throw std::runtime_error(calibration_error);
    }
    if (calibration.model_bytes != 0 &&
        request.model_bytes != 0 &&
        calibration.model_bytes != request.model_bytes) {
      throw std::runtime_error(
          "calibration file was produced for a different model file");
    }
    if (!calibration.execution_provider.empty() &&
        calibration.execution_provider != spec.execution_provider) {
      throw std::runtime_error(
          "calibration file was produced for execution provider '" +
          calibration.execution_provider + "', but current provider is '" +
          spec.execution_provider + "'");
    }
    request.preferred_batch_size = calibration.preferred_batch_size;
    request.calibrated_bytes_per_sample = calibration.calibrated_sample_bytes;
  }
  if (options.has("preferred-batch")) {
    request.preferred_batch_size = options.get_int("preferred-batch", 0);
  }
  if (options.has("calibrated-sample-mb")) {
    request.calibrated_bytes_per_sample =
        static_cast<std::uint64_t>(
            std::max(0, options.get_int("calibrated-sample-mb", 0))) *
        kMiB;
  }
  if (options.has("batch")) {
    request.batch_candidates = {options.get_int("batch", 1)};
    request.preferred_batch_size = 0;
  }
  if (options.has("tile-size")) {
    request.tile_size_candidates = parse_int_list(options.get("tile-size"));
  } else {
    request.tile_size_candidates = {spec.input.width};
  }
  return request;
}

void print_plan(const hiresflow::ExecutionPlan& plan) {
  std::cout << "plan: tile=" << plan.tile_width << "x" << plan.tile_height
            << " overlap=" << plan.overlap_x << "x" << plan.overlap_y
            << " batch=" << plan.batch_size
            << " streams=" << plan.streams
            << " precision=" << hiresflow::precision_name(plan.precision)
            << " classes=" << plan.classes
            << " tiles=" << plan.estimated_tile_count << "\n"
            << "memory: estimated_peak="
            << (plan.estimated_peak_bytes / static_cast<double>(kMiB)) << " MiB"
            << " effective_budget="
            << (plan.effective_budget_bytes / static_cast<double>(kMiB)) << " MiB\n"
            << "budget_source: " << plan.budget_source << "\n"
            << "note: " << plan.planning_note << "\n";
}

void print_backend(const hiresflow::SegmentationBackend& backend) {
  std::cout << "backend: " << backend.spec().name
            << " provider=" << backend.spec().execution_provider << "\n";
}

hiresflow::Normalization normalization_from_options(const Options& options) {
  hiresflow::Normalization normalization;
  const std::string mode = options.get("normalization", "imagenet");
  if (mode == "none") {
    for (int channel = 0; channel < 3; ++channel) {
      normalization.mean[channel] = 0.0F;
      normalization.stddev[channel] = 1.0F;
    }
  } else if (mode != "imagenet") {
    throw std::invalid_argument("normalization must be 'imagenet' or 'none'");
  }
  normalization.rgb_to_bgr = options.has("bgr");
  if (normalization.rgb_to_bgr) {
    std::swap(normalization.mean[0], normalization.mean[2]);
    std::swap(normalization.stddev[0], normalization.stddev[2]);
  }
  return normalization;
}

std::unique_ptr<hiresflow::SegmentationBackend> make_backend(
    const Options& options) {
  const std::string backend_name = options.get("backend", "onnx");
  if (backend_name == "synthetic") {
    return std::make_unique<hiresflow::SyntheticSegmentationBackend>(
        options.get_int("input-width", 512));
  }
  if (backend_name != "onnx") {
    throw std::invalid_argument("unknown backend '" + backend_name + "'");
  }
#ifdef HIRESFLOW_HAS_ORT
  const std::string model_path = options.get("model");
  if (model_path.empty()) {
    throw std::invalid_argument("onnx backend requires --model");
  }
  hiresflow::OnnxBackendOptions backend_options;
  backend_options.prefer_cuda = !options.has("cpu");
  backend_options.enable_profiling = options.has("profile");
  backend_options.classes_override = options.get_int("classes", 0);
  backend_options.input_width_override = options.get_int("input-width", 0);
  backend_options.input_height_override = options.get_int("input-height", 0);
  return std::make_unique<hiresflow::OnnxSegmentationBackend>(
      model_path, backend_options);
#else
  throw std::runtime_error(
      "this binary was built without ONNX Runtime; rebuild with "
      "HIRESFLOW_ENABLE_ORT=ON");
#endif
}

hiresflow::LatencyStats latency_stats(std::vector<double> values) {
  hiresflow::LatencyStats stats;
  if (values.empty()) {
    return stats;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&values](double fraction) {
    if (values.empty()) {
      return 0.0;
    }
    const std::size_t rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    const std::size_t index = std::max<std::size_t>(
        0, std::min(values.size() - 1, rank == 0 ? 0 : rank - 1));
    return values[index];
  };
  stats.mean_ms =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  stats.p50_ms = percentile(0.50);
  stats.p95_ms = percentile(0.95);
  stats.min_ms = values.front();
  stats.max_ms = values.back();
  return stats;
}

int run_plan(const Options& options) {
  const std::string input_path = options.get("input");
  if (input_path.empty()) {
    throw std::invalid_argument("plan requires --input");
  }
  RgbImage image;
  std::string error;
  if (!hiresflow::load_rgb_image(input_path, image, error)) {
    throw std::runtime_error(error);
  }

  std::unique_ptr<hiresflow::SegmentationBackend> backend;
  hiresflow::ModelSpec fallback_spec;
  if (options.get("backend", "onnx") == "synthetic") {
    backend = make_backend(options);
  } else if (options.has("model")) {
    backend = make_backend(options);
  } else {
    fallback_spec.name = "unloaded-model";
    fallback_spec.execution_provider = "unknown";
    fallback_spec.input = ImageSize{
        options.get_int("input-width", 512),
        options.get_int("input-height", 512)};
    fallback_spec.classes = options.get_int("classes", 150);
  }
  const hiresflow::ModelSpec& spec =
      backend ? backend->spec() : fallback_spec;
  if (backend) {
    print_backend(*backend);
  }
  const ExecutionPlan plan =
      hiresflow::create_plan(make_planner_request(options, image, spec));
  print_plan(plan);
  if (options.has("report")) {
    if (!hiresflow::write_plan_json(options.get("report"), plan, error)) {
      throw std::runtime_error(error);
    }
  }
  return 0;
}

int run_pipeline(const Options& options) {
  const std::string input_path = options.get("input");
  const std::string output_path = options.get("output");
  if (input_path.empty() || output_path.empty()) {
    throw std::invalid_argument("run requires --input and --output");
  }

  const auto load_start = std::chrono::steady_clock::now();
  RgbImage image;
  std::string error;
  if (!hiresflow::load_rgb_image(input_path, image, error)) {
    throw std::runtime_error(error);
  }
  const auto load_end = std::chrono::steady_clock::now();

  std::unique_ptr<hiresflow::SegmentationBackend> backend = make_backend(options);
  print_backend(*backend);

  const auto plan_start = std::chrono::steady_clock::now();
  const ExecutionPlan plan = hiresflow::create_plan(
      make_planner_request(options, image, backend->spec()));
  const auto plan_end = std::chrono::steady_clock::now();
  print_plan(plan);

  RunResult result = hiresflow::run_segmentation(
      image, plan, *backend, normalization_from_options(options));
  result.timing.load_ms =
      std::chrono::duration<double, std::milli>(load_end - load_start).count();
  result.timing.plan_ms =
      std::chrono::duration<double, std::milli>(plan_end - plan_start).count();

  const auto write_start = std::chrono::steady_clock::now();
  if (!hiresflow::write_mask_png(output_path, result.mask, error)) {
    throw std::runtime_error(error);
  }
  if (options.has("index-output") &&
      !hiresflow::write_index_mask_png(options.get("index-output"), result.mask, error)) {
    throw std::runtime_error(error);
  }
  const auto write_end = std::chrono::steady_clock::now();
  result.timing.write_ms =
      std::chrono::duration<double, std::milli>(write_end - write_start).count();
  result.timing.total_ms +=
      result.timing.load_ms + result.timing.plan_ms + result.timing.write_ms;

  const std::string report_path =
      options.get("report", output_path + ".report.json");
  if (!hiresflow::write_report_json(
          report_path,
          result,
          input_path,
          output_path,
          backend->spec().name,
          backend->spec().execution_provider,
          options.get("model"),
          error)) {
    throw std::runtime_error(error);
  }

  std::cout << "output: " << output_path << " "
            << result.mask.width << "x" << result.mask.height << "\n"
            << "timing_ms: preprocess=" << result.timing.preprocess_ms
            << " inference=" << result.timing.inference_ms
            << " merge=" << result.timing.merge_ms
            << " total=" << result.timing.total_ms << "\n"
            << "report: " << report_path << "\n";
  return 0;
}

int run_benchmark(const Options& options) {
  const std::string input_path = options.get("input");
  if (input_path.empty()) {
    throw std::invalid_argument("benchmark requires --input");
  }
  const int warmup_runs = options.get_int("warmup", 3);
  const int measured_runs = options.get_int("repeat", 10);
  if (warmup_runs < 0 || measured_runs <= 0) {
    throw std::invalid_argument("warmup must be non-negative and repeat must be positive");
  }

  RgbImage image;
  std::string error;
  if (!hiresflow::load_rgb_image(input_path, image, error)) {
    throw std::runtime_error(error);
  }
  std::unique_ptr<hiresflow::SegmentationBackend> backend = make_backend(options);
  print_backend(*backend);
  const hiresflow::Normalization normalization =
      normalization_from_options(options);
  const hiresflow::ExecutionPlan plan =
      hiresflow::create_plan(make_planner_request(options, image, backend->spec()));
  print_plan(plan);

  for (int iteration = 0; iteration < warmup_runs; ++iteration) {
    const hiresflow::RunResult warmup =
        hiresflow::run_segmentation(image, plan, *backend, normalization);
    (void)warmup;
  }

  std::vector<double> total_latency;
  std::vector<double> inference_latency;
  double preprocess_total = 0.0;
  double merge_total = 0.0;
  double logits_decode_total = 0.0;
  double output_resize_total = 0.0;
  double fusion_total = 0.0;
  bool gpu_sampled = false;
  std::uint64_t gpu_peak = 0;
  int batch_retries = 0;
  int executed_batch_limit = plan.batch_size;
  hiresflow::RunResult last_result;
  for (int iteration = 0; iteration < measured_runs; ++iteration) {
    hiresflow::RunResult result =
        hiresflow::run_segmentation(image, plan, *backend, normalization);
    total_latency.push_back(result.timing.total_ms);
    inference_latency.push_back(result.timing.inference_ms);
    preprocess_total += result.timing.preprocess_ms;
    merge_total += result.timing.merge_ms;
    logits_decode_total += result.timing.logits_decode_ms;
    output_resize_total += result.timing.output_resize_ms;
    fusion_total += result.timing.fusion_ms;
    gpu_sampled = gpu_sampled || result.gpu_memory_sampled;
    gpu_peak = std::max(gpu_peak, result.gpu_peak_used_bytes);
    batch_retries += result.batch_retries;
    executed_batch_limit =
        std::min(executed_batch_limit, result.executed_batch_limit);
    last_result = std::move(result);
  }

  hiresflow::BenchmarkSummary summary;
  summary.plan = plan;
  summary.tiles = last_result.tiles;
  summary.warmup_runs = warmup_runs;
  summary.measured_runs = measured_runs;
  summary.batch_retries = batch_retries;
  summary.executed_batch_limit = executed_batch_limit;
  summary.total = latency_stats(total_latency);
  summary.inference = latency_stats(inference_latency);
  summary.mean_preprocess_ms =
      preprocess_total / static_cast<double>(measured_runs);
  summary.mean_merge_ms = merge_total / static_cast<double>(measured_runs);
  summary.mean_logits_decode_ms =
      logits_decode_total / static_cast<double>(measured_runs);
  summary.mean_output_resize_ms =
      output_resize_total / static_cast<double>(measured_runs);
  summary.mean_fusion_ms =
      fusion_total / static_cast<double>(measured_runs);
  summary.gpu_memory_sampled = gpu_sampled;
  summary.gpu_peak_used_bytes = gpu_peak;

  if (options.has("output") &&
      !hiresflow::write_mask_png(options.get("output"), last_result.mask, error)) {
    throw std::runtime_error(error);
  }
  const std::string report_path =
      options.get("report", "hiresflow-benchmark.json");
  if (!hiresflow::write_benchmark_json(
          report_path,
          summary,
          input_path,
          options.get("model"),
          backend->spec().name,
          backend->spec().execution_provider,
          error)) {
    throw std::runtime_error(error);
  }
  std::cout << "latency_ms: p50=" << summary.total.p50_ms
            << " p95=" << summary.total.p95_ms
            << " mean=" << summary.total.mean_ms << "\n"
            << "inference_ms: p50=" << summary.inference.p50_ms
            << " p95=" << summary.inference.p95_ms << "\n"
            << "gpu_memory: "
            << (summary.gpu_memory_sampled
                    ? std::to_string(summary.gpu_peak_used_bytes / kMiB) + " MiB peak used"
                    : std::string("not sampled"))
            << "\n"
            << "report: " << report_path << "\n";
  return 0;
}

int run_comparison(const Options& options) {
  const std::string input_path = options.get("input");
  const std::string prefix =
      options.get("output", "outputs/comparison");
  if (input_path.empty()) {
    throw std::invalid_argument("compare requires --input");
  }

  RgbImage image;
  std::string error;
  if (!hiresflow::load_rgb_image(input_path, image, error)) {
    throw std::runtime_error(error);
  }
  std::unique_ptr<hiresflow::SegmentationBackend> backend = make_backend(options);
  print_backend(*backend);
  const hiresflow::Normalization normalization =
      normalization_from_options(options);
  const hiresflow::ExecutionPlan plan =
      hiresflow::create_plan(make_planner_request(options, image, backend->spec()));
  print_plan(plan);

  const hiresflow::RunResult tiled =
      hiresflow::run_segmentation(image, plan, *backend, normalization);
  const hiresflow::RunResult downscaled =
      hiresflow::run_downscaled_segmentation(image, *backend, normalization);
  hiresflow::MaskComparison comparison;
  comparison.agreement =
      hiresflow::compare_masks(downscaled.mask, tiled.mask);
  comparison.seams = hiresflow::diagnose_tile_seams(tiled.mask, plan);
  const hiresflow::MaskLabelSummary tiled_labels =
      hiresflow::summarize_mask_labels(tiled.mask, tiled.plan.classes);

  std::optional<hiresflow::SegmentationEvaluation> tiled_evaluation;
  std::optional<hiresflow::SegmentationEvaluation> downscaled_evaluation;
  if (options.has("ground-truth")) {
    hiresflow::MaskImage ground_truth;
    if (!hiresflow::load_index_mask_png(
            options.get("ground-truth"), ground_truth, error)) {
      throw std::runtime_error(error);
    }
    tiled_evaluation = hiresflow::evaluate_segmentation(
        tiled.mask, ground_truth, backend->spec().classes);
    downscaled_evaluation = hiresflow::evaluate_segmentation(
        downscaled.mask, ground_truth, backend->spec().classes);
  }

  const std::filesystem::path prefix_path(prefix);
  if (prefix_path.has_parent_path()) {
    std::filesystem::create_directories(prefix_path.parent_path());
  }
  const std::string tiled_path = prefix + ".tiled.png";
  const std::string downscaled_path = prefix + ".downscaled.png";
  const std::string tiled_index_path = prefix + ".tiled.index.png";
  const std::string downscaled_index_path = prefix + ".downscaled.index.png";
  const std::string report_path = options.get("report", prefix + ".report.json");
  if (!hiresflow::write_mask_png(tiled_path, tiled.mask, error) ||
      !hiresflow::write_mask_png(downscaled_path, downscaled.mask, error) ||
      !hiresflow::write_index_mask_png(tiled_index_path, tiled.mask, error) ||
      !hiresflow::write_index_mask_png(downscaled_index_path, downscaled.mask, error)) {
    throw std::runtime_error(error);
  }
  if (!hiresflow::write_comparison_json(
          report_path,
          comparison,
          tiled,
          downscaled,
          input_path,
          tiled_path,
          downscaled_path,
          options.get("model"),
          backend->spec().name,
          backend->spec().execution_provider,
          tiled_evaluation ? &*tiled_evaluation : nullptr,
          downscaled_evaluation ? &*downscaled_evaluation : nullptr,
          error)) {
    throw std::runtime_error(error);
  }

  std::cout << "agreement: " << comparison.agreement.agreement
            << " (" << comparison.agreement.equal_pixels << "/"
            << comparison.agreement.pixels << " pixels)\n"
            << "tile seams: change_fraction="
            << comparison.seams.seam_change_fraction
            << " non_seam=" << comparison.seams.non_seam_change_fraction << "\n"
            << "label diversity: present=" << tiled_labels.present_labels
            << " dominant=" << tiled_labels.dominant_fraction << "\n"
            << "latency_ms: tiled=" << tiled.timing.total_ms
            << " downscaled=" << downscaled.timing.total_ms << "\n"
            << (tiled_evaluation
                    ? "mIoU: tiled=" + std::to_string(tiled_evaluation->mean_iou) +
                          " downscaled=" +
                          std::to_string(downscaled_evaluation->mean_iou) + "\n"
                    : std::string("ground truth: not provided\n"))
            << "outputs: " << tiled_path << ", " << downscaled_path << "\n"
            << "report: " << report_path << "\n";
  return 0;
}

int run_calibration(const Options& options) {
  const std::string input_path = options.get("input");
  if (input_path.empty()) {
    throw std::invalid_argument("calibrate requires --input");
  }
  const std::vector<int> batches =
      parse_int_list(options.get("batches", "1,2,4,8,16"));
  if (batches.empty()) {
    throw std::invalid_argument("calibration requires at least one batch size");
  }

  RgbImage image;
  std::string error;
  if (!hiresflow::load_rgb_image(input_path, image, error)) {
    throw std::runtime_error(error);
  }
  std::unique_ptr<hiresflow::SegmentationBackend> backend = make_backend(options);
  print_backend(*backend);
  const hiresflow::Normalization normalization =
      normalization_from_options(options);
  const int warmup_runs = options.get_int("warmup", 1);
  const int measured_runs = options.get_int("repeat", 2);
  if (warmup_runs < 0 || measured_runs <= 0) {
    throw std::invalid_argument(
        "calibration warmup must be non-negative and repeat must be positive");
  }

  hiresflow::CalibrationSummary summary;
  summary.execution_provider = backend->spec().execution_provider;
  summary.model_path = options.get("model");
  double best_latency = std::numeric_limits<double>::infinity();
  std::uint64_t maximum_sample_bytes = 0;
  std::uint64_t calibration_baseline_bytes = 0;
#ifdef HIRESFLOW_HAS_CUDA
  if (const auto sample = hiresflow::sample_gpu_memory()) {
    calibration_baseline_bytes = sample->total_bytes - sample->free_bytes;
  }
#endif

  for (const int batch : batches) {
    hiresflow::CalibrationEntry entry;
    entry.requested_batch = batch;
    try {
      PlannerRequest request =
          make_planner_request(options, image, backend->spec());
      request.batch_candidates = {batch};
      request.preferred_batch_size = 0;
      request.calibrated_bytes_per_sample = 0;
      const ExecutionPlan plan = hiresflow::create_plan(request);
      entry.executed_batch = plan.batch_size;
      print_plan(plan);

      for (int iteration = 0; iteration < warmup_runs; ++iteration) {
        const hiresflow::RunResult warmup =
            hiresflow::run_segmentation(image, plan, *backend, normalization);
        (void)warmup;
      }
      std::vector<double> total_latencies;
      std::vector<double> inference_latencies;
      for (int iteration = 0; iteration < measured_runs; ++iteration) {
        const hiresflow::RunResult result =
            hiresflow::run_segmentation(image, plan, *backend, normalization);
        total_latencies.push_back(result.timing.total_ms);
        inference_latencies.push_back(result.timing.inference_ms);
        entry.executed_batch = result.executed_batch_limit;
        entry.gpu_peak_used_bytes =
            std::max(entry.gpu_peak_used_bytes, result.gpu_peak_used_bytes);
        const std::uint64_t baseline = calibration_baseline_bytes > 0
            ? calibration_baseline_bytes
            : result.gpu_baseline_used_bytes;
        if (result.gpu_memory_sampled &&
            result.gpu_peak_used_bytes > baseline &&
            result.executed_batch_limit > 0) {
          maximum_sample_bytes = std::max(
              maximum_sample_bytes,
              (result.gpu_peak_used_bytes - baseline) /
                  static_cast<std::uint64_t>(result.executed_batch_limit));
        }
      }
      const hiresflow::LatencyStats total = latency_stats(total_latencies);
      const hiresflow::LatencyStats inference =
          latency_stats(inference_latencies);
      entry.total_p50_ms = total.p50_ms;
      entry.total_p95_ms = total.p95_ms;
      entry.inference_p50_ms = inference.p50_ms;
      entry.success = true;
      if (entry.requested_batch == entry.executed_batch &&
          total.p50_ms < best_latency) {
        best_latency = total.p50_ms;
        summary.selected_batch = entry.requested_batch;
      }
    } catch (const std::exception& exception) {
      entry.error = exception.what();
    }
    std::cout << "batch=" << entry.requested_batch
              << " executed=" << entry.executed_batch
              << " success=" << (entry.success ? "true" : "false")
              << " p50_ms=" << entry.total_p50_ms
              << " inference_p50_ms=" << entry.inference_p50_ms
              << " peak_mib="
              << entry.gpu_peak_used_bytes / static_cast<double>(kMiB) << "\n";
    summary.entries.push_back(std::move(entry));
  }

  if (summary.selected_batch <= 0) {
    throw std::runtime_error("no calibrated batch completed successfully");
  }
  summary.calibrated_sample_bytes = maximum_sample_bytes;
  const std::string report_path =
      options.get("report", "outputs/calibration.json");
  const std::string config_path =
      options.get("config", "outputs/calibration.conf");
  if (!hiresflow::write_calibration_json(report_path, summary, error) ||
      !hiresflow::write_calibration_config(
          config_path,
          summary,
          file_size(options.get("model")),
          error)) {
    throw std::runtime_error(error);
  }
  std::cout << "selected_batch=" << summary.selected_batch
            << " calibrated_sample_mib="
            << summary.calibrated_sample_bytes / static_cast<double>(kMiB)
            << "\n"
            << "report: " << report_path << "\n"
            << "config: " << config_path << "\n";
  return 0;
}

int run_evaluation(const Options& options) {
  const std::string prediction_path = options.get("prediction");
  const std::string ground_truth_path = options.get("ground-truth");
  if (prediction_path.empty() || ground_truth_path.empty()) {
    throw std::invalid_argument("evaluate requires --prediction and --ground-truth");
  }
  hiresflow::MaskImage prediction;
  hiresflow::MaskImage ground_truth;
  std::string error;
  if (!hiresflow::load_index_mask_png(prediction_path, prediction, error) ||
      !hiresflow::load_index_mask_png(ground_truth_path, ground_truth, error)) {
    throw std::runtime_error(error);
  }
  const int classes = options.get_int("classes", 0);
  if (classes <= 0) {
    throw std::invalid_argument("evaluate requires --classes");
  }
  const hiresflow::SegmentationEvaluation evaluation =
      hiresflow::evaluate_segmentation(prediction, ground_truth, classes);
  const std::string report_path =
      options.get("report", "hiresflow-evaluation.json");
  if (!hiresflow::write_evaluation_json(
          report_path,
          evaluation,
          prediction_path,
          ground_truth_path,
          error)) {
    throw std::runtime_error(error);
  }
  std::cout << "mIoU: " << evaluation.mean_iou
            << " pixel_accuracy: " << evaluation.pixel_accuracy
            << " pixels: " << evaluation.pixels << "\n"
            << "report: " << report_path << "\n";
  return 0;
}

bool is_supported_image(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

int run_directory(const Options& options) {
  const std::string input_directory = options.get("input-dir");
  const std::string output_directory = options.get("output-dir");
  if (input_directory.empty() || output_directory.empty()) {
    throw std::invalid_argument("run-dir requires --input-dir and --output-dir");
  }
  const std::filesystem::path input_root(input_directory);
  const std::filesystem::path output_root(output_directory);
  if (!std::filesystem::is_directory(input_root)) {
    throw std::invalid_argument("input directory does not exist: " + input_directory);
  }
  std::filesystem::create_directories(output_root);

  std::vector<std::filesystem::path> image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(input_root)) {
    if (entry.is_regular_file() && is_supported_image(entry.path())) {
      image_paths.push_back(entry.path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  if (image_paths.empty()) {
    throw std::runtime_error("no PNG or JPEG images found in " + input_directory);
  }

  std::unique_ptr<hiresflow::SegmentationBackend> backend = make_backend(options);
  print_backend(*backend);
  const hiresflow::Normalization normalization =
      normalization_from_options(options);
  hiresflow::BatchSummary summary;
  summary.backend_name = backend->spec().name;
  summary.execution_provider = backend->spec().execution_provider;
  summary.total_images = static_cast<int>(image_paths.size());
  summary.warmup_runs = options.get_int("warmup", 0);
  if (summary.warmup_runs < 0) {
    throw std::invalid_argument("run-dir warmup must be non-negative");
  }
  if (summary.warmup_runs > 0) {
    RgbImage warmup_image;
    std::string warmup_error;
    if (hiresflow::load_rgb_image(image_paths.front().string(), warmup_image, warmup_error)) {
      const ExecutionPlan warmup_plan = hiresflow::create_plan(
          make_planner_request(options, warmup_image, backend->spec()));
      for (int iteration = 0; iteration < summary.warmup_runs; ++iteration) {
        const hiresflow::RunResult warmup =
            hiresflow::run_segmentation(
                warmup_image, warmup_plan, *backend, normalization);
        (void)warmup;
      }
      std::cout << "warmup: " << summary.warmup_runs << " run(s)\n";
    } else {
      std::cerr << "warmup skipped: " << warmup_error << "\n";
    }
  }
  const auto wall_start = std::chrono::steady_clock::now();
  std::vector<double> total_latencies;
  std::vector<double> inference_latencies;
  double preprocess_total = 0.0;
  double merge_total = 0.0;

  for (const std::filesystem::path& input_path : image_paths) {
    hiresflow::BatchImageSummary image_summary;
    image_summary.input_path = input_path.string();
    const std::string stem = (output_root / input_path.stem()).string();
    image_summary.output_path = stem + ".mask.png";
    try {
      const auto load_start = std::chrono::steady_clock::now();
      RgbImage image;
      std::string error;
      if (!hiresflow::load_rgb_image(input_path.string(), image, error)) {
        throw std::runtime_error(error);
      }
      const auto load_end = std::chrono::steady_clock::now();

      const auto plan_start = std::chrono::steady_clock::now();
      const ExecutionPlan plan =
          hiresflow::create_plan(make_planner_request(options, image, backend->spec()));
      const auto plan_end = std::chrono::steady_clock::now();
      hiresflow::RunResult result =
          hiresflow::run_segmentation(image, plan, *backend, normalization);
      result.timing.load_ms =
          std::chrono::duration<double, std::milli>(load_end - load_start).count();
      result.timing.plan_ms =
          std::chrono::duration<double, std::milli>(plan_end - plan_start).count();

      const auto write_start = std::chrono::steady_clock::now();
      const std::string index_path = stem + ".index.png";
      const std::string report_path = stem + ".report.json";
      if (!hiresflow::write_mask_png(image_summary.output_path, result.mask, error) ||
          !hiresflow::write_index_mask_png(index_path, result.mask, error)) {
        throw std::runtime_error(error);
      }
      const auto write_end = std::chrono::steady_clock::now();
      result.timing.write_ms =
          std::chrono::duration<double, std::milli>(write_end - write_start).count();
      result.timing.total_ms +=
          result.timing.load_ms + result.timing.plan_ms + result.timing.write_ms;
      if (!hiresflow::write_report_json(
              report_path,
              result,
              input_path.string(),
              image_summary.output_path,
              backend->spec().name,
              backend->spec().execution_provider,
              options.get("model"),
              error)) {
        throw std::runtime_error(error);
      }
      image_summary.success = true;
      image_summary.tiles = result.tiles;
      image_summary.batch_size = plan.batch_size;
      image_summary.batch_retries = result.batch_retries;
      image_summary.load_ms = result.timing.load_ms;
      image_summary.preprocess_ms = result.timing.preprocess_ms;
      image_summary.inference_ms = result.timing.inference_ms;
      image_summary.merge_ms = result.timing.merge_ms;
      image_summary.write_ms = result.timing.write_ms;
      image_summary.total_ms = result.timing.total_ms;
      image_summary.gpu_peak_used_bytes = result.gpu_peak_used_bytes;
      summary.total_tiles += result.tiles;
      total_latencies.push_back(result.timing.total_ms);
      inference_latencies.push_back(result.timing.inference_ms);
      preprocess_total += result.timing.preprocess_ms;
      merge_total += result.timing.merge_ms;
      ++summary.successful_images;
      std::cout << "ok: " << input_path.filename().string()
                << " total_ms=" << image_summary.total_ms
                << " tiles=" << image_summary.tiles << "\n";
    } catch (const std::exception& exception) {
      image_summary.error = exception.what();
      ++summary.failed_images;
      std::cerr << "failed: " << input_path.filename().string()
                << ": " << image_summary.error << "\n";
    }
    summary.images.push_back(std::move(image_summary));
  }

  const auto wall_end = std::chrono::steady_clock::now();
  summary.wall_time_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  summary.total_latency = latency_stats(total_latencies);
  summary.inference_latency = latency_stats(inference_latencies);
  if (summary.successful_images > 0) {
    summary.mean_preprocess_ms =
        preprocess_total / static_cast<double>(summary.successful_images);
    summary.mean_merge_ms =
        merge_total / static_cast<double>(summary.successful_images);
  }
  const std::string report_path =
      options.get("report", (output_root / "summary.json").string());
  std::string error;
  if (!hiresflow::write_batch_summary_json(report_path, summary, error)) {
    throw std::runtime_error(error);
  }
  std::cout << "images: " << summary.successful_images << " succeeded, "
            << summary.failed_images << " failed, "
            << summary.wall_time_ms << " ms wall time\n"
            << "summary: " << report_path << "\n";
  return summary.failed_images == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.command == "help" || options.command == "--help" ||
        options.command == "-h") {
      print_usage();
      return 0;
    }
    if (options.command == "plan") {
      return run_plan(options);
    }
    if (options.command == "run") {
      return run_pipeline(options);
    }
    if (options.command == "benchmark") {
      return run_benchmark(options);
    }
    if (options.command == "compare") {
      return run_comparison(options);
    }
    if (options.command == "calibrate") {
      return run_calibration(options);
    }
    if (options.command == "evaluate") {
      return run_evaluation(options);
    }
    if (options.command == "run-dir") {
      return run_directory(options);
    }
    print_usage();
    throw std::invalid_argument("unknown command '" + options.command + "'");
  } catch (const std::exception& exception) {
    std::cerr << "hiresflow: " << exception.what() << "\n";
    return 1;
  }
}
