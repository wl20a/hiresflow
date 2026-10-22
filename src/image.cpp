#include "hiresflow/core.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace hiresflow {
namespace {

float bilinear_coordinate(int destination_index, int source_size, int destination_size) {
  return (static_cast<float>(destination_index) + 0.5F) *
             static_cast<float>(source_size) /
             static_cast<float>(destination_size) -
         0.5F;
}

int clamp_int(int value, int low, int high) {
  return std::max(low, std::min(value, high));
}

std::uint8_t class_red(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  return static_cast<std::uint8_t>(48U + value % 192U);
}

std::uint8_t class_green(std::uint32_t value) {
  value ^= value >> 13U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return static_cast<std::uint8_t>(48U + value % 192U);
}

std::uint8_t class_blue(std::uint32_t value) {
  value ^= value >> 17U;
  value *= 0x9e3779b1U;
  value ^= value >> 15U;
  return static_cast<std::uint8_t>(48U + value % 192U);
}

}  // namespace

bool load_rgb_image(const std::string& path, RgbImage& image, std::string& error) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* decoded = stbi_load(path.c_str(), &width, &height, &channels, 3);
  if (decoded == nullptr) {
    const char* reason = stbi_failure_reason();
    error = "failed to load image '" + path + "': " +
            (reason == nullptr ? std::string("unknown decoder error") : std::string(reason));
    return false;
  }

  image.width = width;
  image.height = height;
  const std::size_t bytes =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
  image.pixels.assign(decoded, decoded + bytes);
  stbi_image_free(decoded);
  return true;
}

bool write_rgb_png(const std::string& path, const RgbImage& image, std::string& error) {
  if (image.empty()) {
    error = "cannot write an empty RGB image";
    return false;
  }

  const int result = stbi_write_png(
      path.c_str(),
      image.width,
      image.height,
      3,
      image.pixels.data(),
      image.width * 3);
  if (result == 0) {
    error = "failed to write PNG '" + path + "'";
    return false;
  }
  return true;
}

bool write_mask_png(const std::string& path, const MaskImage& mask, std::string& error) {
  if (mask.width <= 0 || mask.height <= 0 ||
      mask.labels.size() !=
          static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height)) {
    error = "mask dimensions and label buffer do not match";
    return false;
  }

  RgbImage colored;
  colored.width = mask.width;
  colored.height = mask.height;
  colored.pixels.resize(mask.labels.size() * 3U);

  for (std::size_t i = 0; i < mask.labels.size(); ++i) {
    const std::uint32_t label = mask.labels[i];
    colored.pixels[i * 3U] = class_red(label);
    colored.pixels[i * 3U + 1U] = class_green(label);
    colored.pixels[i * 3U + 2U] = class_blue(label);
  }
  return write_rgb_png(path, colored, error);
}

bool load_index_mask_png(const std::string& path, MaskImage& mask, std::string& error) {
  int width = 0;
  int height = 0;
  int channels = 0;
  if (stbi_is_16_bit(path.c_str()) != 0) {
    stbi_us* decoded = stbi_load_16(path.c_str(), &width, &height, &channels, 1);
    if (decoded == nullptr) {
      error = "failed to load 16-bit index mask '" + path + "'";
      return false;
    }
    if (channels != 1) {
      stbi_image_free(decoded);
      error = "index mask must have one channel";
      return false;
    }
    mask.width = width;
    mask.height = height;
    const std::size_t pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    mask.labels.assign(decoded, decoded + pixels);
    stbi_image_free(decoded);
    return true;
  }

  stbi_uc* decoded = stbi_load(path.c_str(), &width, &height, &channels, 1);
  if (decoded == nullptr) {
    error = "failed to load index mask '" + path + "'";
    return false;
  }
  if (channels != 1) {
    stbi_image_free(decoded);
    error = "index mask must have one channel; palette or RGB masks require an explicit mapping";
    return false;
  }
  mask.width = width;
  mask.height = height;
  const std::size_t pixels =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  mask.labels.resize(pixels);
  for (std::size_t index = 0; index < pixels; ++index) {
    mask.labels[index] = decoded[index];
  }
  stbi_image_free(decoded);
  return true;
}

bool write_index_mask_png(
    const std::string& path,
    const MaskImage& mask,
    std::string& error) {
  if (mask.width <= 0 || mask.height <= 0 ||
      mask.labels.size() !=
          static_cast<std::size_t>(mask.width) * static_cast<std::size_t>(mask.height)) {
    error = "index mask dimensions and label buffer do not match";
    return false;
  }
  std::vector<std::uint8_t> bytes(mask.labels.size());
  for (std::size_t index = 0; index < mask.labels.size(); ++index) {
    if (mask.labels[index] > 255) {
      error = "8-bit index PNG output supports at most 256 classes";
      return false;
    }
    bytes[index] = static_cast<std::uint8_t>(mask.labels[index]);
  }
  if (stbi_write_png(
          path.c_str(),
          mask.width,
          mask.height,
          1,
          bytes.data(),
          mask.width) == 0) {
    error = "failed to write index mask PNG '" + path + "'";
    return false;
  }
  return true;
}

RgbImage crop_rgb(const RgbImage& image, const Tile& tile) {
  RgbImage output;
  if (tile.width <= 0 || tile.height <= 0 || tile.x < 0 || tile.y < 0 ||
      tile.x + tile.width > image.width || tile.y + tile.height > image.height) {
    return output;
  }

  output.width = tile.width;
  output.height = tile.height;
  output.pixels.resize(
      static_cast<std::size_t>(tile.width) * static_cast<std::size_t>(tile.height) * 3U);

  const std::size_t row_bytes = static_cast<std::size_t>(tile.width) * 3U;
  for (int y = 0; y < tile.height; ++y) {
    const std::size_t source_offset =
        (static_cast<std::size_t>(tile.y + y) * static_cast<std::size_t>(image.width) +
         static_cast<std::size_t>(tile.x)) *
        3U;
    const std::size_t destination_offset =
        static_cast<std::size_t>(y) * row_bytes;
    std::copy_n(
        image.pixels.begin() + static_cast<std::ptrdiff_t>(source_offset),
        row_bytes,
        output.pixels.begin() + static_cast<std::ptrdiff_t>(destination_offset));
  }
  return output;
}

RgbImage resize_rgb_bilinear(const RgbImage& image, int width, int height) {
  RgbImage output;
  if (image.empty() || width <= 0 || height <= 0) {
    return output;
  }

  output.width = width;
  output.height = height;
  output.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

  for (int destination_y = 0; destination_y < height; ++destination_y) {
    const float source_y = bilinear_coordinate(destination_y, image.height, height);
    const int y0 = clamp_int(static_cast<int>(std::floor(source_y)), 0, image.height - 1);
    const int y1 = clamp_int(y0 + 1, 0, image.height - 1);
    const float fy = std::max(0.0F, std::min(1.0F, source_y - std::floor(source_y)));

    for (int destination_x = 0; destination_x < width; ++destination_x) {
      const float source_x = bilinear_coordinate(destination_x, image.width, width);
      const int x0 = clamp_int(static_cast<int>(std::floor(source_x)), 0, image.width - 1);
      const int x1 = clamp_int(x0 + 1, 0, image.width - 1);
      const float fx = std::max(0.0F, std::min(1.0F, source_x - std::floor(source_x)));

      for (int channel = 0; channel < 3; ++channel) {
        const auto pixel = [&](int x, int y) {
          return static_cast<float>(
              image.pixels[(static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
                            static_cast<std::size_t>(x)) *
                               3U +
                           static_cast<std::size_t>(channel)]);
        };
        const float top = pixel(x0, y0) * (1.0F - fx) + pixel(x1, y0) * fx;
        const float bottom = pixel(x0, y1) * (1.0F - fx) + pixel(x1, y1) * fx;
        const float value = top * (1.0F - fy) + bottom * fy;
        const std::size_t destination_index =
            (static_cast<std::size_t>(destination_y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(destination_x)) *
                3U +
            static_cast<std::size_t>(channel);
        output.pixels[destination_index] =
            static_cast<std::uint8_t>(std::lround(std::max(0.0F, std::min(255.0F, value))));
      }
    }
  }
  return output;
}

std::vector<std::uint16_t> resize_mask_nearest(
    const std::vector<std::uint16_t>& labels,
    int source_width,
    int source_height,
    int target_width,
    int target_height) {
  if (target_width <= 0 || target_height <= 0 || source_width <= 0 || source_height <= 0 ||
      labels.size() !=
          static_cast<std::size_t>(source_width) * static_cast<std::size_t>(source_height)) {
    return {};
  }
  if (source_width == target_width && source_height == target_height) {
    return labels;
  }

  std::vector<std::uint16_t> output(
      static_cast<std::size_t>(target_width) * static_cast<std::size_t>(target_height));
  for (int y = 0; y < target_height; ++y) {
    const int source_y =
        clamp_int(static_cast<int>(static_cast<long long>(y) * source_height / target_height),
                  0,
                  source_height - 1);
    for (int x = 0; x < target_width; ++x) {
      const int source_x =
          clamp_int(static_cast<int>(static_cast<long long>(x) * source_width / target_width),
                    0,
                    source_width - 1);
      output[static_cast<std::size_t>(y) * static_cast<std::size_t>(target_width) +
             static_cast<std::size_t>(x)] =
          labels[static_cast<std::size_t>(source_y) * static_cast<std::size_t>(source_width) +
                 static_cast<std::size_t>(source_x)];
    }
  }
  return output;
}

std::vector<float> resize_scalar_bilinear(
    const std::vector<float>& values,
    int source_width,
    int source_height,
    int target_width,
    int target_height) {
  if (target_width <= 0 || target_height <= 0 || source_width <= 0 || source_height <= 0 ||
      values.size() !=
          static_cast<std::size_t>(source_width) * static_cast<std::size_t>(source_height)) {
    return {};
  }
  if (source_width == target_width && source_height == target_height) {
    return values;
  }

  std::vector<float> output(
      static_cast<std::size_t>(target_width) * static_cast<std::size_t>(target_height));
  for (int destination_y = 0; destination_y < target_height; ++destination_y) {
    const float source_y = bilinear_coordinate(destination_y, source_height, target_height);
    const int y0 = clamp_int(static_cast<int>(std::floor(source_y)), 0, source_height - 1);
    const int y1 = clamp_int(y0 + 1, 0, source_height - 1);
    const float fy = std::max(0.0F, std::min(1.0F, source_y - std::floor(source_y)));
    for (int destination_x = 0; destination_x < target_width; ++destination_x) {
      const float source_x = bilinear_coordinate(destination_x, source_width, target_width);
      const int x0 = clamp_int(static_cast<int>(std::floor(source_x)), 0, source_width - 1);
      const int x1 = clamp_int(x0 + 1, 0, source_width - 1);
      const float fx = std::max(0.0F, std::min(1.0F, source_x - std::floor(source_x)));
      const auto value = [&](int x, int y) {
        return values[static_cast<std::size_t>(y) * static_cast<std::size_t>(source_width) +
                      static_cast<std::size_t>(x)];
      };
      const float top = value(x0, y0) * (1.0F - fx) + value(x1, y0) * fx;
      const float bottom = value(x0, y1) * (1.0F - fx) + value(x1, y1) * fx;
      output[static_cast<std::size_t>(destination_y) * static_cast<std::size_t>(target_width) +
             static_cast<std::size_t>(destination_x)] =
          top * (1.0F - fy) + bottom * fy;
    }
  }
  return output;
}

}  // namespace hiresflow
