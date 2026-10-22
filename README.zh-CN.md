# HiResFlow

**在单张 GPU 上运行现有高分辨率语义分割模型。**

[English](README.md) | 简体中文

HiResFlow 是一个面向现有 FP32 ONNX 语义分割模型的 C++ 推理运行时。它会读取模型约束，根据可用显存规划 batch，在目标 GPU 上校准实际吞吐，执行切片推理，将结果恢复到原图坐标，并生成可复现的性能报告。

> **状态：** v0.1 技术预览。API、CLI 参数、报告格式和内部接口仍可能变化。

HiResFlow 不做超分辨率、不生成新像素、不训练模型，也不重写 CUDA 图像处理库。当前图片解码和预处理仍在 CPU 上完成。

## 它解决什么问题

高分辨率图片通常不能直接输入固定尺寸的视觉模型。手工缩图会丢失小目标，手工切片又需要处理显存、batch、重叠、坐标映射和结果合并。

HiResFlow 负责把“图片、模型、GPU”连接成一条可执行且可测量的管线：

```text
图片 → 模型约束检查 → 显存规划 → batch 校准
     → 切片推理 → 坐标恢复 → 重叠融合 → 原尺寸结果与报告
```

## 当前能力

- PNG/JPG 输入，彩色 mask 和 8 位类别索引 PNG 输出。
- ONNX Runtime CUDA/CPU 后端。
- 模型输入尺寸、类别数和固定 batch 约束读取。
- 可用显存查询、安全余量和显存预算规划。
- 基于实测延迟的 batch 校准与 OOM 自动降级。
- 完整覆盖原图的均匀切片、坐标映射和重叠融合。
- P50/P95、分阶段耗时、峰值显存和目录批处理报告。
- 与整图缩放基线的对比、接缝诊断和 ground truth mIoU。
- C++ CLI 和最小 Python 封装。

## 实测结果

测试环境：RTX 5070 Laptop 8 GB、ONNX Runtime 1.31 CUDA 12、FCN ResNet-50 FP32。

1538×905、8 个 tile，校准后 batch=1：

- 端到端 P50：327 ms
- 推理 P50：215 ms
- 峰值显存：702 MiB

6720×4480 NASA 样例：

- 512 切片、64 重叠：6.40 秒，保留 18 个类别
- 整图缩放到 512：0.30 秒，保留 4 个类别
- 两种结果像素一致率：93.66%

该结果说明切片保留了更多小目标，但**不代表准确率更高**。当前仍缺少带标注高分辨率数据集上的完整 mIoU 结论。

## 快速开始

当前验证平台为 Linux x86_64：

```bash
git clone https://github.com/wl20a/hiresflow.git
cd hiresflow
bash scripts/quickstart.sh
```

脚本会下载固定版本的依赖、模型和公开样例，编译并运行测试；检测到 NVIDIA GPU 时会自动执行校准和 CUDA smoke test。

运行单张图片：

```bash
./build/hiresflow run \
  --backend onnx \
  --model work/models/fcn-resnet50-12.onnx \
  --calibration outputs/host-gpu-calibration.conf \
  --input path/to/image.jpg \
  --output outputs/mask.png \
  --index-output outputs/mask.index.png \
  --report outputs/report.json \
  --input-width 512 \
  --input-height 512 \
  --classes 21
```

批量处理目录：

```bash
bash scripts/host_gpu_batch.sh path/to/images outputs/batch
```

## Python 调用

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
export HIRESFLOW_BIN="$PWD/build/hiresflow"
```

```python
import hiresflow

report = hiresflow.segment(
    "image.jpg",
    "mask.png",
    "work/models/fcn-resnet50-12.onnx",
    input_size=(512, 512),
    classes=21,
    calibration_path="outputs/host-gpu-calibration.conf",
)
print(report["timing_ms"])
```

当前 Python 接口是对 C++ CLI 的封装，尚无原生扩展或可直接安装的 wheel。

## 已知限制

- 当前执行器只有单 CUDA Stream。
- CPU 预处理、融合和 PNG 编码尚未与下一张图片的 GPU 推理重叠。
- 尚未支持自动 FP16、TensorRT、GPU 解码和 GPU 预处理。
- 融合使用 top-1/top-2 margin 置信度，不是完整的全类别概率融合。
- 仅支持语义分割，尚无检测和 OCR 结果合并。
- 尚无图片、mask、检测框和关键点同步切图。
- 仅验证 Linux x86_64，尚无预编译包、Docker 和 GPU CI。
- Alpha 透明背景和 EXIF 旋转尚未完整处理。
- 尚未完成带 ground truth 的高分辨率精度评测。

## 后续计划

1. 通过异步输出和 CPU/GPU 流水线降低大图尾延迟。
2. 增加 FP16 校准和 TensorRT 实验后端。
3. 接入 nvImageCodec、CV-CUDA 等 GPU 图像处理组件。
4. 改进多类别重叠融合和接缝质量。
5. 增加图片、mask、检测框和关键点同步切图。
6. 增加检测和 OCR 合并适配器。
7. 提供带标注基准、预编译版本、容器镜像和原生 Python wheel。

## 反馈与贡献

项目仍在持续开发中。后续会继续改进执行效率、模型兼容性、结果质量、安装体验和文档。

欢迎提交 Issue、复现实验、性能数据和 Pull Request。遇到不兼容的 ONNX 模型时，请附上 GPU、驱动、ONNX Runtime 版本、完整命令、JSON 报告和最小复现。

开发与测试说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

项目采用 Apache License 2.0，详见 [LICENSE](LICENSE)。

第三方依赖和公开样例的授权说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
