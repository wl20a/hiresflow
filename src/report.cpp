#include "hiresflow/core.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace hiresflow {
namespace {

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        output << character;
        break;
    }
  }
  return output.str();
}

}  // namespace

bool write_report_json(
    const std::string& path,
    const RunResult& result,
    const std::string& input_path,
    const std::string& output_path,
    const std::string& backend_name,
    const std::string& execution_provider,
    const std::string& model_path,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open report path '" + path + "'";
    return false;
  }

  const ExecutionPlan& plan = result.plan;
  const StageTiming& timing = result.timing;
  report << std::fixed << std::setprecision(3);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"input\": \"" << json_escape(input_path) << "\",\n";
  report << "  \"output\": \"" << json_escape(output_path) << "\",\n";
  report << "  \"model\": \"" << json_escape(model_path) << "\",\n";
  report << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
  report << "  \"execution_provider\": \"" << json_escape(execution_provider) << "\",\n";
  report << "  \"image\": {\"width\": " << plan.image.width
         << ", \"height\": " << plan.image.height << "},\n";
  report << "  \"plan\": {\n";
  report << "    \"tile_width\": " << plan.tile_width << ",\n";
  report << "    \"tile_height\": " << plan.tile_height << ",\n";
  report << "    \"overlap_x\": " << plan.overlap_x << ",\n";
  report << "    \"overlap_y\": " << plan.overlap_y << ",\n";
  report << "    \"batch_size\": " << plan.batch_size << ",\n";
  report << "    \"streams\": " << plan.streams << ",\n";
  report << "    \"precision\": \"" << precision_name(plan.precision) << "\",\n";
  report << "    \"classes\": " << plan.classes << ",\n";
  report << "    \"estimated_tile_count\": " << plan.estimated_tile_count << ",\n";
  report << "    \"estimated_peak_bytes\": " << plan.estimated_peak_bytes << ",\n";
  report << "    \"effective_budget_bytes\": " << plan.effective_budget_bytes << ",\n";
  report << "    \"budget_source\": \"" << json_escape(plan.budget_source) << "\",\n";
  report << "    \"planning_note\": \"" << json_escape(plan.planning_note) << "\"\n";
  report << "  },\n";
  report << "  \"tiles_executed\": " << result.tiles << ",\n";
  report << "  \"adaptive_batch\": {\"retries\": " << result.batch_retries
         << ", \"executed_batch_limit\": " << result.executed_batch_limit << "},\n";
  report << "  \"gpu_memory\": {\"sampled\": "
         << (result.gpu_memory_sampled ? "true" : "false")
         << ", \"baseline_used_bytes\": " << result.gpu_baseline_used_bytes
         << ", \"peak_used_bytes\": " << result.gpu_peak_used_bytes << "},\n";
  report << "  \"timing_ms\": {\n";
  report << "    \"load\": " << timing.load_ms << ",\n";
  report << "    \"plan\": " << timing.plan_ms << ",\n";
  report << "    \"preprocess\": " << timing.preprocess_ms << ",\n";
  report << "    \"inference\": " << timing.inference_ms << ",\n";
  report << "    \"merge\": " << timing.merge_ms << ",\n";
  report << "    \"logits_decode\": " << timing.logits_decode_ms << ",\n";
  report << "    \"output_resize\": " << timing.output_resize_ms << ",\n";
  report << "    \"fusion\": " << timing.fusion_ms << ",\n";
  report << "    \"write\": " << timing.write_ms << ",\n";
  report << "    \"total\": " << timing.total_ms << "\n";
  report << "  }\n";
  report << "}\n";

  if (!report) {
    error = "failed while writing report '" + path + "'";
    return false;
  }
  return true;
}

bool write_plan_json(
    const std::string& path,
    const ExecutionPlan& plan,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open plan path '" + path + "'";
    return false;
  }
  report << std::fixed << std::setprecision(3);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"image\": {\"width\": " << plan.image.width
         << ", \"height\": " << plan.image.height << "},\n";
  report << "  \"model_input\": {\"width\": " << plan.model_input.width
         << ", \"height\": " << plan.model_input.height << "},\n";
  report << "  \"tile_width\": " << plan.tile_width << ",\n";
  report << "  \"tile_height\": " << plan.tile_height << ",\n";
  report << "  \"overlap_x\": " << plan.overlap_x << ",\n";
  report << "  \"overlap_y\": " << plan.overlap_y << ",\n";
  report << "  \"batch_size\": " << plan.batch_size << ",\n";
  report << "  \"streams\": " << plan.streams << ",\n";
  report << "  \"classes\": " << plan.classes << ",\n";
  report << "  \"precision\": \"" << precision_name(plan.precision) << "\",\n";
  report << "  \"estimated_tile_count\": " << plan.estimated_tile_count << ",\n";
  report << "  \"estimated_peak_bytes\": " << plan.estimated_peak_bytes << ",\n";
  report << "  \"effective_budget_bytes\": " << plan.effective_budget_bytes << ",\n";
  report << "  \"budget_source\": \"" << json_escape(plan.budget_source) << "\",\n";
  report << "  \"planning_note\": \"" << json_escape(plan.planning_note) << "\"\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing plan '" + path + "'";
    return false;
  }
  return true;
}

bool write_benchmark_json(
    const std::string& path,
    const BenchmarkSummary& summary,
    const std::string& input_path,
    const std::string& model_path,
    const std::string& backend_name,
    const std::string& execution_provider,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open benchmark report path '" + path + "'";
    return false;
  }

  const auto write_latency = [&report](const char* name, const LatencyStats& stats) {
    report << "  \"" << name << "\": {\"mean\": " << stats.mean_ms
           << ", \"p50\": " << stats.p50_ms
           << ", \"p95\": " << stats.p95_ms
           << ", \"min\": " << stats.min_ms
           << ", \"max\": " << stats.max_ms << "},\n";
  };

  report << std::fixed << std::setprecision(3);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"input\": \"" << json_escape(input_path) << "\",\n";
  report << "  \"model\": \"" << json_escape(model_path) << "\",\n";
  report << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
  report << "  \"execution_provider\": \"" << json_escape(execution_provider) << "\",\n";
  report << "  \"warmup_runs\": " << summary.warmup_runs << ",\n";
  report << "  \"measured_runs\": " << summary.measured_runs << ",\n";
  report << "  \"adaptive_batch\": {\"retries\": " << summary.batch_retries
         << ", \"executed_batch_limit\": " << summary.executed_batch_limit << "},\n";
  report << "  \"tiles_per_run\": " << summary.tiles << ",\n";
  write_latency("total_latency_ms", summary.total);
  write_latency("inference_latency_ms", summary.inference);
  report << "  \"mean_preprocess_ms\": " << summary.mean_preprocess_ms << ",\n";
  report << "  \"mean_merge_ms\": " << summary.mean_merge_ms << ",\n";
  report << "  \"mean_logits_decode_ms\": " << summary.mean_logits_decode_ms << ",\n";
  report << "  \"mean_output_resize_ms\": " << summary.mean_output_resize_ms << ",\n";
  report << "  \"mean_fusion_ms\": " << summary.mean_fusion_ms << ",\n";
  report << "  \"gpu_memory\": {\"sampled\": "
         << (summary.gpu_memory_sampled ? "true" : "false")
         << ", \"peak_used_bytes\": " << summary.gpu_peak_used_bytes << "},\n";
  report << "  \"plan\": {\"tile_width\": " << summary.plan.tile_width
         << ", \"tile_height\": " << summary.plan.tile_height
         << ", \"overlap_x\": " << summary.plan.overlap_x
         << ", \"overlap_y\": " << summary.plan.overlap_y
         << ", \"batch_size\": " << summary.plan.batch_size
         << ", \"streams\": " << summary.plan.streams
         << ", \"precision\": \"" << precision_name(summary.plan.precision) << "\"}\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing benchmark report '" + path + "'";
    return false;
  }
  return true;
}

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
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open comparison report path '" + path + "'";
    return false;
  }

  report << std::fixed << std::setprecision(6);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"input\": \"" << json_escape(input_path) << "\",\n";
  report << "  \"model\": \"" << json_escape(model_path) << "\",\n";
  report << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
  report << "  \"execution_provider\": \"" << json_escape(execution_provider) << "\",\n";
  report << "  \"tiled_output\": \"" << json_escape(tiled_output_path) << "\",\n";
  report << "  \"downscaled_output\": \"" << json_escape(downscaled_output_path) << "\",\n";
  report << "  \"metrics\": {\n";
  report << "    \"mask_agreement\": " << comparison.agreement.agreement << ",\n";
  report << "    \"equal_pixels\": " << comparison.agreement.equal_pixels << ",\n";
  report << "    \"different_pixels\": "
         << comparison.agreement.pixels - comparison.agreement.equal_pixels << ",\n";
  report << "    \"tile_seam_edges\": " << comparison.seams.seam_edges << ",\n";
  report << "    \"tile_seam_change_fraction\": "
         << comparison.seams.seam_change_fraction << ",\n";
  report << "    \"non_seam_edge_change_fraction\": "
         << comparison.seams.non_seam_change_fraction << "\n";
  report << "  },\n";
  const MaskLabelSummary tiled_labels =
      summarize_mask_labels(tiled.mask, tiled.plan.classes);
  const MaskLabelSummary downscaled_labels =
      summarize_mask_labels(downscaled.mask, downscaled.plan.classes);
  report << "  \"label_coverage\": {\n";
  report << "    \"tiled_present_labels\": " << tiled_labels.present_labels << ",\n";
  report << "    \"tiled_dominant_fraction\": " << tiled_labels.dominant_fraction << ",\n";
  report << "    \"downscaled_present_labels\": "
         << downscaled_labels.present_labels << ",\n";
  report << "    \"downscaled_dominant_fraction\": "
         << downscaled_labels.dominant_fraction << "\n";
  report << "  },\n";
  report << "  \"interpretation\": \"These are consistency diagnostics, not "
            "ground-truth accuracy. A high agreement does not prove correctness.\",\n";
  report << "  \"warnings\": [";
  const bool low_diversity =
      tiled_labels.present_labels <= 1 || downscaled_labels.present_labels <= 1;
  if (low_diversity) {
    report << "\"At least one output contains only one class; this input cannot "
              "demonstrate segmentation quality differences.\"";
  }
  report << "],\n";
  const auto write_evaluation = [&report](
                                    const char* name,
                                    const SegmentationEvaluation* evaluation) {
    if (evaluation == nullptr) {
      return;
    }
    report << "  \"" << name << "\": {\"mean_iou\": " << evaluation->mean_iou
           << ", \"pixel_accuracy\": " << evaluation->pixel_accuracy
           << ", \"class_iou\": [";
    for (std::size_t index = 0; index < evaluation->class_iou.size(); ++index) {
      if (index > 0) {
        report << ", ";
      }
      report << evaluation->class_iou[index];
    }
    report << "]},\n";
  };
  write_evaluation("tiled_ground_truth", tiled_evaluation);
  write_evaluation("downscaled_ground_truth", downscaled_evaluation);
  report << "  \"tiled\": {\n";
  report << "    \"timing_ms\": {\"total\": " << tiled.timing.total_ms
         << ", \"preprocess\": " << tiled.timing.preprocess_ms
         << ", \"inference\": " << tiled.timing.inference_ms
         << ", \"merge\": " << tiled.timing.merge_ms << "},\n";
  report << "    \"tiles\": " << tiled.tiles << ",\n";
  report << "    \"plan\": {\"tile_width\": " << tiled.plan.tile_width
         << ", \"tile_height\": " << tiled.plan.tile_height
         << ", \"overlap_x\": " << tiled.plan.overlap_x
         << ", \"overlap_y\": " << tiled.plan.overlap_y
         << ", \"batch_size\": " << tiled.plan.batch_size
         << ", \"streams\": " << tiled.plan.streams
         << ", \"precision\": \"" << precision_name(tiled.plan.precision) << "\"},\n";
  report << "    \"gpu_memory\": {\"sampled\": "
         << (tiled.gpu_memory_sampled ? "true" : "false")
         << ", \"peak_used_bytes\": " << tiled.gpu_peak_used_bytes << "}\n";
  report << "  },\n";
  report << "  \"downscaled\": {\n";
  report << "    \"timing_ms\": {\"total\": " << downscaled.timing.total_ms
         << ", \"preprocess\": " << downscaled.timing.preprocess_ms
         << ", \"inference\": " << downscaled.timing.inference_ms
         << ", \"merge\": " << downscaled.timing.merge_ms << "},\n";
  report << "    \"tiles\": " << downscaled.tiles << ",\n";
  report << "    \"gpu_memory\": {\"sampled\": "
         << (downscaled.gpu_memory_sampled ? "true" : "false")
         << ", \"peak_used_bytes\": " << downscaled.gpu_peak_used_bytes << "}\n";
  report << "  }\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing comparison report '" + path + "'";
    return false;
  }
  return true;
}

bool write_evaluation_json(
    const std::string& path,
    const SegmentationEvaluation& evaluation,
    const std::string& prediction_path,
    const std::string& ground_truth_path,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open evaluation report path '" + path + "'";
    return false;
  }
  report << std::fixed << std::setprecision(6);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"prediction\": \"" << json_escape(prediction_path) << "\",\n";
  report << "  \"ground_truth\": \"" << json_escape(ground_truth_path) << "\",\n";
  report << "  \"classes\": " << evaluation.classes << ",\n";
  report << "  \"pixels\": " << evaluation.pixels << ",\n";
  report << "  \"pixel_accuracy\": " << evaluation.pixel_accuracy << ",\n";
  report << "  \"mean_iou\": " << evaluation.mean_iou << ",\n";
  report << "  \"class_iou\": [";
  for (std::size_t index = 0; index < evaluation.class_iou.size(); ++index) {
    if (index > 0) {
      report << ", ";
    }
    report << evaluation.class_iou[index];
  }
  report << "]\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing evaluation report '" + path + "'";
    return false;
  }
  return true;
}

bool write_batch_summary_json(
    const std::string& path,
    const BatchSummary& summary,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open batch summary path '" + path + "'";
    return false;
  }
  report << std::fixed << std::setprecision(3);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"backend\": \"" << json_escape(summary.backend_name) << "\",\n";
  report << "  \"execution_provider\": \""
         << json_escape(summary.execution_provider) << "\",\n";
  report << "  \"total_images\": " << summary.total_images << ",\n";
  report << "  \"successful_images\": " << summary.successful_images << ",\n";
  report << "  \"failed_images\": " << summary.failed_images << ",\n";
  report << "  \"warmup_runs\": " << summary.warmup_runs << ",\n";
  report << "  \"total_tiles\": " << summary.total_tiles << ",\n";
  report << "  \"wall_time_ms\": " << summary.wall_time_ms << ",\n";
  report << "  \"total_latency_ms\": {\"mean\": " << summary.total_latency.mean_ms
         << ", \"p50\": " << summary.total_latency.p50_ms
         << ", \"p95\": " << summary.total_latency.p95_ms
         << ", \"min\": " << summary.total_latency.min_ms
         << ", \"max\": " << summary.total_latency.max_ms << "},\n";
  report << "  \"inference_latency_ms\": {\"mean\": "
         << summary.inference_latency.mean_ms
         << ", \"p50\": " << summary.inference_latency.p50_ms
         << ", \"p95\": " << summary.inference_latency.p95_ms
         << ", \"min\": " << summary.inference_latency.min_ms
         << ", \"max\": " << summary.inference_latency.max_ms << "},\n";
  report << "  \"mean_preprocess_ms\": " << summary.mean_preprocess_ms << ",\n";
  report << "  \"mean_merge_ms\": " << summary.mean_merge_ms << ",\n";
  report << "  \"images\": [\n";
  for (std::size_t index = 0; index < summary.images.size(); ++index) {
    const BatchImageSummary& image = summary.images[index];
    report << "    {\"input\": \"" << json_escape(image.input_path)
           << "\", \"output\": \"" << json_escape(image.output_path)
           << "\", \"success\": " << (image.success ? "true" : "false")
           << ", \"error\": \"" << json_escape(image.error)
           << "\", \"tiles\": " << image.tiles
           << ", \"batch_size\": " << image.batch_size
           << ", \"batch_retries\": " << image.batch_retries
           << ", \"load_ms\": " << image.load_ms
           << ", \"preprocess_ms\": " << image.preprocess_ms
           << ", \"inference_ms\": " << image.inference_ms
           << ", \"merge_ms\": " << image.merge_ms
           << ", \"write_ms\": " << image.write_ms
           << ", \"total_ms\": " << image.total_ms
           << ", \"gpu_peak_used_bytes\": " << image.gpu_peak_used_bytes << "}";
    if (index + 1 < summary.images.size()) {
      report << ",";
    }
    report << "\n";
  }
  report << "  ]\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing batch summary '" + path + "'";
    return false;
  }
  return true;
}

}  // namespace hiresflow
