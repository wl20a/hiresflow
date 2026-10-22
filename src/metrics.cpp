#include "hiresflow/core.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace hiresflow {

SegmentationEvaluation evaluate_segmentation(
    const MaskImage& prediction,
    const MaskImage& ground_truth,
    int classes) {
  if (classes <= 0) {
    throw std::invalid_argument("evaluation requires a positive class count");
  }
  if (prediction.width != ground_truth.width ||
      prediction.height != ground_truth.height ||
      prediction.labels.size() != ground_truth.labels.size()) {
    throw std::invalid_argument("prediction and ground truth masks have different dimensions");
  }

  SegmentationEvaluation evaluation;
  evaluation.classes = classes;
  evaluation.pixels = prediction.labels.size();
  evaluation.class_iou.assign(static_cast<std::size_t>(classes), 0.0);
  evaluation.intersections.assign(static_cast<std::size_t>(classes), 0);
  evaluation.unions.assign(static_cast<std::size_t>(classes), 0);
  std::vector<std::uint64_t> prediction_counts(static_cast<std::size_t>(classes), 0);
  std::vector<std::uint64_t> truth_counts(static_cast<std::size_t>(classes), 0);
  std::vector<std::uint64_t> intersection_counts(static_cast<std::size_t>(classes), 0);

  for (std::size_t index = 0; index < prediction.labels.size(); ++index) {
    const int predicted = static_cast<int>(prediction.labels[index]);
    const int truth = static_cast<int>(ground_truth.labels[index]);
    if (predicted >= classes || truth >= classes) {
      throw std::runtime_error("mask contains a class outside the evaluation range");
    }
    ++prediction_counts[static_cast<std::size_t>(predicted)];
    ++truth_counts[static_cast<std::size_t>(truth)];
    if (predicted == truth) {
      ++intersection_counts[static_cast<std::size_t>(truth)];
      ++evaluation.correct_pixels;
    }
  }

  double iou_sum = 0.0;
  std::size_t present_classes = 0;
  for (int class_index = 0; class_index < classes; ++class_index) {
    const std::size_t index = static_cast<std::size_t>(class_index);
    evaluation.intersections[index] = intersection_counts[index];
    evaluation.unions[index] =
        prediction_counts[index] + truth_counts[index] - intersection_counts[index];
    if (evaluation.unions[index] > 0) {
      evaluation.class_iou[index] =
          static_cast<double>(evaluation.intersections[index]) /
          static_cast<double>(evaluation.unions[index]);
      iou_sum += evaluation.class_iou[index];
      ++present_classes;
    }
  }
  evaluation.mean_iou =
      present_classes == 0 ? 0.0 : iou_sum / static_cast<double>(present_classes);
  evaluation.pixel_accuracy = evaluation.pixels == 0
      ? 0.0
      : static_cast<double>(evaluation.correct_pixels) /
            static_cast<double>(evaluation.pixels);
  return evaluation;
}

}  // namespace hiresflow
