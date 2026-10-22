#pragma once

#include "hiresflow/types.hpp"

#include <memory>
#include <string>

namespace hiresflow {

struct OnnxBackendOptions {
  bool prefer_cuda = true;
  bool enable_profiling = false;
  int classes_override = 0;
  int input_width_override = 0;
  int input_height_override = 0;
};

class OnnxSegmentationBackend final : public SegmentationBackend {
 public:
  OnnxSegmentationBackend(const std::string& model_path, const OnnxBackendOptions& options);
  ~OnnxSegmentationBackend() override;

  OnnxSegmentationBackend(const OnnxSegmentationBackend&) = delete;
  OnnxSegmentationBackend& operator=(const OnnxSegmentationBackend&) = delete;

  const ModelSpec& spec() const override;
  std::vector<ModelLogits> infer(const std::vector<TileTensor>& batch) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hiresflow
