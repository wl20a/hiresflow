#include "hiresflow/core.hpp"

#ifdef HIRESFLOW_HAS_CUDA
#include "hiresflow/gpu_memory.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace hiresflow {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool is_out_of_memory_error(const std::exception& error) {
  std::string message = error.what();
  std::transform(message.begin(), message.end(), message.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return message.find("out of memory") != std::string::npos ||
         message.find("out-of-memory") != std::string::npos ||
         message.find("cuda_error_out_of_memory") != std::string::npos ||
         message.find("failed to allocate memory") != std::string::npos ||
         message.find("failed to allocate") != std::string::npos ||
         message.find("bfc_arena") != std::string::npos;
}

TileTensor preprocess_tile(
    const RgbImage& image,
    const Tile& tile,
    const ModelSpec& spec,
    const Normalization& normalization) {
  RgbImage crop = crop_rgb(image, tile);
  if (crop.empty()) {
    throw std::runtime_error("failed to crop an image tile");
  }
  if (crop.width != spec.input.width || crop.height != spec.input.height) {
    crop = resize_rgb_bilinear(crop, spec.input.width, spec.input.height);
  }

  TileTensor tensor;
  tensor.tile = tile;
  tensor.width = spec.input.width;
  tensor.height = spec.input.height;
  tensor.channels = 3;
  const std::size_t plane =
      static_cast<std::size_t>(tensor.width) * static_cast<std::size_t>(tensor.height);
  tensor.data.resize(plane * 3U);

  for (int y = 0; y < tensor.height; ++y) {
    for (int x = 0; x < tensor.width; ++x) {
      const std::size_t pixel =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(tensor.width) +
          static_cast<std::size_t>(x);
      for (int channel = 0; channel < 3; ++channel) {
        const int source_channel =
            normalization.rgb_to_bgr ? 2 - channel : channel;
        const float value =
            static_cast<float>(crop.pixels[pixel * 3U + static_cast<std::size_t>(source_channel)]) /
            255.0F;
        tensor.data[static_cast<std::size_t>(channel) * plane + pixel] =
            (value - normalization.mean[channel]) /
            std::max(1.0e-6F, normalization.stddev[channel]);
      }
    }
  }
  return tensor;
}

struct ClassMap {
  std::vector<std::uint16_t> labels;
  std::vector<float> confidence;
};

ClassMap logits_to_class_map(const ModelLogits& logits) {
  if (logits.classes <= 0 || logits.width <= 0 || logits.height <= 0 ||
      logits.data.size() !=
          static_cast<std::size_t>(logits.classes) *
              static_cast<std::size_t>(logits.width) *
              static_cast<std::size_t>(logits.height)) {
    throw std::runtime_error("model returned invalid logits");
  }

  const std::size_t pixels =
      static_cast<std::size_t>(logits.width) * static_cast<std::size_t>(logits.height);
  ClassMap map;
  map.labels.resize(pixels);
  map.confidence.resize(pixels);

  for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
    float maximum = -std::numeric_limits<float>::infinity();
    float second = -std::numeric_limits<float>::infinity();
    int maximum_class = 0;
    for (int class_index = 0; class_index < logits.classes; ++class_index) {
      const float value =
          logits.data[static_cast<std::size_t>(class_index) * pixels + pixel];
      if (value > maximum) {
        second = maximum;
        maximum = value;
        maximum_class = class_index;
      } else if (value > second) {
        second = value;
      }
    }

    map.labels[pixel] = static_cast<std::uint16_t>(maximum_class);
    const float margin =
        std::isfinite(second) ? maximum - second : 16.0F;
    map.confidence[pixel] = 1.0F / (1.0F + std::exp(-std::max(0.0F, margin)));
  }
  return map;
}

}  // namespace

RunResult run_segmentation(
    const RgbImage& image,
    const ExecutionPlan& plan,
    SegmentationBackend& backend,
    const Normalization& normalization) {
  if (image.width != plan.image.width || image.height != plan.image.height) {
    throw std::invalid_argument("image dimensions do not match the execution plan");
  }
  if (plan.batch_size <= 0) {
    throw std::invalid_argument("execution plan has an invalid batch size");
  }

  RunResult result;
  result.plan = plan;
  const auto total_start = Clock::now();
  const std::vector<Tile> tiles = make_tiles(
      image.width,
      image.height,
      plan.tile_width,
      plan.tile_height,
      plan.overlap_x,
      plan.overlap_y);
  result.tiles = static_cast<int>(tiles.size());
  result.executed_batch_limit = plan.batch_size;
  SegmentationFuser fuser(image.width, image.height, plan.classes);

#ifdef HIRESFLOW_HAS_CUDA
  if (const auto sample = sample_gpu_memory()) {
    result.gpu_memory_sampled = true;
    result.gpu_baseline_used_bytes = sample->total_bytes - sample->free_bytes;
    result.gpu_peak_used_bytes = result.gpu_baseline_used_bytes;
  }
#endif

  std::size_t begin = 0;
  int active_batch_limit = plan.batch_size;
  while (begin < tiles.size()) {
    const std::size_t end =
        std::min(tiles.size(), begin + static_cast<std::size_t>(active_batch_limit));
    std::vector<TileTensor> batch;
    batch.reserve(end - begin);

    const auto preprocess_start = Clock::now();
    for (std::size_t index = begin; index < end; ++index) {
      batch.push_back(preprocess_tile(image, tiles[index], backend.spec(), normalization));
    }
    const auto preprocess_end = Clock::now();
    result.timing.preprocess_ms += elapsed_ms(preprocess_start, preprocess_end);

    const auto inference_start = Clock::now();
    std::vector<ModelLogits> outputs;
    try {
      outputs = backend.infer(batch);
    } catch (const std::exception& inference_error) {
      if (!is_out_of_memory_error(inference_error) || active_batch_limit <= 1) {
        throw;
      }
      active_batch_limit = std::max(1, active_batch_limit / 2);
      result.executed_batch_limit =
          std::min(result.executed_batch_limit, active_batch_limit);
      ++result.batch_retries;
      continue;
    }
    const auto inference_end = Clock::now();
    result.timing.inference_ms += elapsed_ms(inference_start, inference_end);

#ifdef HIRESFLOW_HAS_CUDA
    if (const auto sample = sample_gpu_memory()) {
      result.gpu_memory_sampled = true;
      const std::uint64_t used = sample->total_bytes - sample->free_bytes;
      result.gpu_peak_used_bytes = std::max(result.gpu_peak_used_bytes, used);
    }
#endif

    if (outputs.size() != batch.size()) {
      throw std::runtime_error("backend returned a different number of outputs than inputs");
    }

    const auto merge_start = Clock::now();
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto decode_start = Clock::now();
      const ClassMap class_map = logits_to_class_map(outputs[index]);
      const auto decode_end = Clock::now();
      result.timing.logits_decode_ms += elapsed_ms(decode_start, decode_end);
      const Tile& tile = batch[index].tile;
      const auto resize_start = Clock::now();
      const std::vector<std::uint16_t> source_labels = resize_mask_nearest(
          class_map.labels,
          outputs[index].width,
          outputs[index].height,
          tile.width,
          tile.height);
      const std::vector<float> source_confidence = resize_scalar_bilinear(
          class_map.confidence,
          outputs[index].width,
          outputs[index].height,
          tile.width,
          tile.height);
      const auto resize_end = Clock::now();
      result.timing.output_resize_ms += elapsed_ms(resize_start, resize_end);
      const auto fusion_start = Clock::now();
      fuser.add(
          tile,
          source_labels,
          source_confidence,
          tile.width,
          tile.height);
      const auto fusion_end = Clock::now();
      result.timing.fusion_ms += elapsed_ms(fusion_start, fusion_end);
    }
    const auto merge_end = Clock::now();
    result.timing.merge_ms += elapsed_ms(merge_start, merge_end);
    begin = end;
  }

  result.mask = fuser.finish();
  const auto total_end = Clock::now();
  result.timing.total_ms = elapsed_ms(total_start, total_end);
  return result;
}

RunResult run_downscaled_segmentation(
    const RgbImage& image,
    SegmentationBackend& backend,
    const Normalization& normalization) {
  if (image.empty()) {
    throw std::invalid_argument("whole-image baseline requires a non-empty image");
  }

  RunResult result;
  result.plan.image = image.size();
  result.plan.model_input = backend.spec().input;
  result.plan.tile_width = image.width;
  result.plan.tile_height = image.height;
  result.plan.overlap_x = 0;
  result.plan.overlap_y = 0;
  result.plan.batch_size = 1;
  result.plan.streams = 1;
  result.plan.classes = backend.spec().classes;
  result.plan.precision = Precision::kFloat32;
  result.plan.estimated_tile_count = 1;
  result.plan.planning_note = "whole-image downscale baseline";
  result.tiles = 1;
  const auto total_start = Clock::now();

#ifdef HIRESFLOW_HAS_CUDA
  if (const auto sample = sample_gpu_memory()) {
    result.gpu_memory_sampled = true;
    result.gpu_baseline_used_bytes = sample->total_bytes - sample->free_bytes;
    result.gpu_peak_used_bytes = result.gpu_baseline_used_bytes;
  }
#endif

  const Tile whole_tile{
      0,
      0,
      0,
      image.width,
      image.height};
  const auto preprocess_start = Clock::now();
  const TileTensor tensor =
      preprocess_tile(image, whole_tile, backend.spec(), normalization);
  const auto preprocess_end = Clock::now();
  result.timing.preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);

  const auto inference_start = Clock::now();
  const std::vector<ModelLogits> outputs = backend.infer({tensor});
  const auto inference_end = Clock::now();
  result.timing.inference_ms = elapsed_ms(inference_start, inference_end);

#ifdef HIRESFLOW_HAS_CUDA
  if (const auto sample = sample_gpu_memory()) {
    result.gpu_memory_sampled = true;
    const std::uint64_t used = sample->total_bytes - sample->free_bytes;
    result.gpu_peak_used_bytes = std::max(result.gpu_peak_used_bytes, used);
  }
#endif

  if (outputs.size() != 1) {
    throw std::runtime_error("backend did not return one baseline output");
  }
  const auto merge_start = Clock::now();
  const auto decode_start = Clock::now();
  const ClassMap class_map = logits_to_class_map(outputs[0]);
  const auto decode_end = Clock::now();
  result.timing.logits_decode_ms = elapsed_ms(decode_start, decode_end);
  result.mask.width = image.width;
  result.mask.height = image.height;
  const auto resize_start = Clock::now();
  result.mask.labels = resize_mask_nearest(
      class_map.labels,
      outputs[0].width,
      outputs[0].height,
      image.width,
      image.height);
  const auto resize_end = Clock::now();
  result.timing.output_resize_ms = elapsed_ms(resize_start, resize_end);
  result.mask.confidence = resize_scalar_bilinear(
      class_map.confidence,
      outputs[0].width,
      outputs[0].height,
      image.width,
      image.height);
  const auto merge_end = Clock::now();
  result.timing.merge_ms = elapsed_ms(merge_start, merge_end);
  const auto total_end = Clock::now();
  result.timing.total_ms = elapsed_ms(total_start, total_end);
  return result;
}

}  // namespace hiresflow
