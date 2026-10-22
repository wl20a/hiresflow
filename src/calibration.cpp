#include "hiresflow/core.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace hiresflow {
namespace {

std::string escape_json(const std::string& value) {
  std::ostringstream output;
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      output << '\\' << character;
    } else if (character == '\n') {
      output << "\\n";
    } else {
      output << character;
    }
  }
  return output.str();
}

bool parse_u64(const std::string& value, std::uint64_t& output) {
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    return false;
  }
  output = parsed;
  return true;
}

}  // namespace

bool write_calibration_json(
    const std::string& path,
    const CalibrationSummary& summary,
    std::string& error) {
  std::ofstream report(path);
  if (!report) {
    error = "failed to open calibration report '" + path + "'";
    return false;
  }
  report << std::fixed << std::setprecision(3);
  report << "{\n";
  report << "  \"schema_version\": \"0.1\",\n";
  report << "  \"model\": \"" << escape_json(summary.model_path) << "\",\n";
  report << "  \"execution_provider\": \""
         << escape_json(summary.execution_provider) << "\",\n";
  report << "  \"selected_batch\": " << summary.selected_batch << ",\n";
  report << "  \"calibrated_sample_bytes\": "
         << summary.calibrated_sample_bytes << ",\n";
  report << "  \"results\": [\n";
  for (std::size_t index = 0; index < summary.entries.size(); ++index) {
    const CalibrationEntry& entry = summary.entries[index];
    report << "    {\"requested_batch\": " << entry.requested_batch
           << ", \"executed_batch\": " << entry.executed_batch
           << ", \"success\": " << (entry.success ? "true" : "false")
           << ", \"error\": \"" << escape_json(entry.error)
           << "\", \"total_p50_ms\": " << entry.total_p50_ms
           << ", \"total_p95_ms\": " << entry.total_p95_ms
           << ", \"inference_p50_ms\": " << entry.inference_p50_ms
           << ", \"gpu_peak_used_bytes\": " << entry.gpu_peak_used_bytes << "}";
    if (index + 1 < summary.entries.size()) {
      report << ",";
    }
    report << "\n";
  }
  report << "  ]\n";
  report << "}\n";
  if (!report) {
    error = "failed while writing calibration report '" + path + "'";
    return false;
  }
  return true;
}

bool write_calibration_config(
    const std::string& path,
    const CalibrationSummary& summary,
    std::uint64_t model_bytes,
    std::string& error) {
  std::ofstream config(path);
  if (!config) {
    error = "failed to open calibration config '" + path + "'";
    return false;
  }
  config << "schema_version=0.1\n";
  config << "preferred_batch=" << summary.selected_batch << "\n";
  config << "calibrated_sample_bytes=" << summary.calibrated_sample_bytes << "\n";
  config << "model_bytes=" << model_bytes << "\n";
  config << "execution_provider=" << summary.execution_provider << "\n";
  if (!config) {
    error = "failed while writing calibration config '" + path + "'";
    return false;
  }
  return true;
}

bool load_calibration_config(
    const std::string& path,
    CalibrationConfig& calibration,
    std::string& error) {
  std::ifstream config(path);
  if (!config) {
    error = "failed to open calibration config '" + path + "'";
    return false;
  }
  std::string line;
  int line_number = 0;
  while (std::getline(config, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
      error = "invalid calibration config line " + std::to_string(line_number);
      return false;
    }
    const std::string key = line.substr(0, separator);
    const std::string value = line.substr(separator + 1);
    std::uint64_t parsed = 0;
    if (key == "schema_version") {
      continue;
    }
    if (key == "execution_provider") {
      calibration.execution_provider = value;
      continue;
    }
    if (!parse_u64(value, parsed)) {
      error = "invalid integer in calibration config line " +
              std::to_string(line_number);
      return false;
    }
    if (key == "preferred_batch") {
      calibration.preferred_batch_size = static_cast<int>(parsed);
    } else if (key == "calibrated_sample_bytes") {
      calibration.calibrated_sample_bytes = parsed;
    } else if (key == "model_bytes") {
      calibration.model_bytes = parsed;
    }
  }
  if (calibration.preferred_batch_size <= 0) {
    error = "calibration config has no valid preferred_batch";
    return false;
  }
  return true;
}

}  // namespace hiresflow
