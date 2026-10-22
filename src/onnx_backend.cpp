#include "hiresflow/onnx_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hiresflow {

class OnnxSegmentationBackend::Impl {
 public:
  Impl(const std::string& model_path, const OnnxBackendOptions& options)
      : env_(ORT_LOGGING_LEVEL_ERROR, "hiresflow"),
        allocator_() {
    if (model_path.empty()) {
      throw std::invalid_argument("ONNX model path is empty");
    }

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);
    if (options.enable_profiling) {
      session_options.EnableProfiling("hiresflow_ort_profile");
      profiling_enabled_ = true;
    }

    bool cuda_enabled = false;
    if (options.prefer_cuda) {
      const std::vector<std::string> providers = Ort::GetAvailableProviders();
      const bool cuda_available =
          std::find(
              providers.begin(),
              providers.end(),
              "CUDAExecutionProvider") != providers.end();
      if (cuda_available) {
        try {
          OrtCUDAProviderOptions cuda_options{};
          cuda_options.device_id = 0;
          session_options.AppendExecutionProvider_CUDA(cuda_options);
          cuda_enabled = true;
        } catch (const Ort::Exception&) {
          cuda_enabled = false;
        }
      }
    }

    try {
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception&) {
      if (!cuda_enabled) {
        throw;
      }
      Ort::SessionOptions cpu_options;
      cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      cpu_options.SetIntraOpNumThreads(1);
      if (options.enable_profiling) {
        cpu_options.EnableProfiling("hiresflow_ort_profile_cpu");
      }
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), cpu_options);
      cuda_enabled = false;
    }
    input_name_ = owned_name(session_->GetInputNameAllocated(0, allocator_));
    output_name_ = owned_name(session_->GetOutputNameAllocated(0, allocator_));
    inspect_model(options, cuda_enabled);
  }

  ~Impl() {
    if (!session_ || !profiling_enabled_) {
      return;
    }
    try {
      const Ort::AllocatedStringPtr profile =
          session_->EndProfilingAllocated(allocator_);
      const std::string profile_path =
          profile.get() == nullptr ? std::string() : std::string(profile.get());
      if (!profile_path.empty()) {
        std::cerr << "ORT profile: " << profile_path << "\n";
      }
    } catch (const Ort::Exception&) {
      // Profiling is diagnostic-only; never fail shutdown because of it.
    }
  }

  std::string owned_name(Ort::AllocatedStringPtr allocated) const {
    const char* value = allocated.get();
    return value == nullptr ? std::string() : std::string(value);
  }

  void inspect_model(const OnnxBackendOptions& options, bool cuda_enabled) {
    const Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    const auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    if (input_tensor_info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error(
          "the first HiResFlow backend requires an FP32 ONNX input tensor");
    }
    const std::vector<std::int64_t> input_shape = input_tensor_info.GetShape();
    if (input_shape.size() != 4 || input_shape[1] != 3) {
      throw std::runtime_error(
          "expected NCHW model input with three image channels");
    }

    spec_.name = "onnx:" + input_name_;
    spec_.execution_provider =
        cuda_enabled ? "CUDAExecutionProvider" : "CPUExecutionProvider";
    spec_.input.width =
        options.input_width_override > 0
            ? options.input_width_override
            : static_cast<int>(input_shape[3]);
    spec_.input.height =
        options.input_height_override > 0
            ? options.input_height_override
            : static_cast<int>(input_shape[2]);
    if (spec_.input.width <= 0 || spec_.input.height <= 0) {
      throw std::runtime_error(
          "the model has dynamic spatial input dimensions; pass both "
          "--input-width and --input-height");
    }
    if (input_shape[0] > 0) {
      spec_.fixed_batch_size = static_cast<int>(input_shape[0]);
    }

    const Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
    const auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    if (output_tensor_info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error(
          "the first HiResFlow backend requires FP32 segmentation output");
    }
    const std::vector<std::int64_t> output_shape = output_tensor_info.GetShape();
    if (options.classes_override > 0) {
      spec_.classes = options.classes_override;
    } else if (output_shape.size() == 4 && output_shape[1] > 0) {
      spec_.classes = static_cast<int>(output_shape[1]);
    } else if (output_shape.size() == 3 && output_shape[0] > 0) {
      spec_.classes = static_cast<int>(output_shape[0]);
    } else {
      throw std::runtime_error(
          "unable to infer output classes from the ONNX model; pass --classes");
    }
    if (spec_.classes <= 0 || spec_.classes > 65535) {
      throw std::runtime_error("unsupported segmentation class count");
    }
  }

  std::vector<ModelLogits> infer(const std::vector<TileTensor>& batch) {
    if (batch.empty()) {
      return {};
    }
    const std::size_t expected_values =
        static_cast<std::size_t>(spec_.input.width) *
        static_cast<std::size_t>(spec_.input.height) * 3U;
    for (const TileTensor& tile : batch) {
      if (tile.width != spec_.input.width || tile.height != spec_.input.height ||
          tile.data.size() != expected_values) {
        throw std::invalid_argument("ONNX batch tensor shape does not match model input");
      }
    }

    std::vector<float> input_values;
    input_values.reserve(expected_values * batch.size());
    for (const TileTensor& tile : batch) {
      input_values.insert(input_values.end(), tile.data.begin(), tile.data.end());
    }
    const std::array<std::int64_t, 4> input_shape{
        static_cast<std::int64_t>(batch.size()),
        3,
        static_cast<std::int64_t>(spec_.input.height),
        static_cast<std::int64_t>(spec_.input.width)};
    const Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_values.data(),
        input_values.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
      throw std::runtime_error("ONNX model returned no tensor output");
    }

    const Ort::Value& output_tensor = output_tensors[0];
    const auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("ONNX model output tensor is not FP32");
    }
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    if (output_shape.size() != 4 || output_shape[0] != static_cast<std::int64_t>(batch.size())) {
      throw std::runtime_error("expected ONNX output shape [N,C,H,W]");
    }

    const int classes = static_cast<int>(output_shape[1]);
    const int height = static_cast<int>(output_shape[2]);
    const int width = static_cast<int>(output_shape[3]);
    if (classes <= 0 || height <= 0 || width <= 0) {
      throw std::runtime_error("ONNX output contains dynamic or invalid dimensions");
    }
    if (classes != spec_.classes) {
      throw std::runtime_error("ONNX output classes do not match the model specification");
    }

    const float* data = output_tensor.GetTensorData<float>();
    const std::size_t values_per_sample =
        static_cast<std::size_t>(classes) *
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width);
    std::vector<ModelLogits> outputs;
    outputs.reserve(batch.size());
    for (std::size_t index = 0; index < batch.size(); ++index) {
      ModelLogits logits;
      logits.classes = classes;
      logits.width = width;
      logits.height = height;
      logits.data.assign(
          data + index * values_per_sample,
          data + (index + 1) * values_per_sample);
      outputs.push_back(std::move(logits));
    }
    return outputs;
  }

  ModelSpec spec_;
  Ort::Env env_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  bool profiling_enabled_ = false;
};

OnnxSegmentationBackend::OnnxSegmentationBackend(
    const std::string& model_path,
    const OnnxBackendOptions& options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}

OnnxSegmentationBackend::~OnnxSegmentationBackend() = default;

const ModelSpec& OnnxSegmentationBackend::spec() const {
  return impl_->spec_;
}

std::vector<ModelLogits> OnnxSegmentationBackend::infer(
    const std::vector<TileTensor>& batch) {
  return impl_->infer(batch);
}

}  // namespace hiresflow
