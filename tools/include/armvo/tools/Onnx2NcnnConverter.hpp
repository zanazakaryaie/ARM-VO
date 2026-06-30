#pragma once

#include <string>
#include <array>
#include <optional>
#include <memory>
#include <cstdint>

#include <armvo/Types.hpp>

namespace armvo::tools
{

struct Int8QuantizationConfig
{
    /// Directory containing representative calibration images
    std::string calibImagesDir;
    /// Per-channel input mean values
    std::array<float, 3> mean{0.3257f * 255.f,
                              0.3690f * 255.f,
                              0.3223f * 255.f};
    /// Per-channel input normalization values
    std::array<float, 3> norm{1.f / (0.2112f * 255.f),
                              1.f / (0.2148f * 255.f),
                              1.f / (0.2115f * 255.f)};

    /// Model input shape as width, height, and channels
    std::array<uint32_t, 3> modelInputShape{640, 320, 3};
    /// Pixel format expected by the model input
    armvo::PixelFormat modelInputPixelFormat = armvo::PixelFormat::RGB;
};

struct ModelConversionConfig
{
    std::string onnxFilepath;
    /// Directory where model.param, model.bin, and model.shape are written
    std::string outputDir;
    /// Model input height
    uint32_t inputHeight;
    /// Model input width
    uint32_t inputWidth;
    bool optimize = true;
    std::optional<Int8QuantizationConfig> quantization = std::nullopt;
};

enum class ModelConversionStatus : uint8_t
{
    SUCCESS,
    INVALID_ONNX_FILEPATH,
    CONVERSION_FAILED,
    OPTIMIZATION_FAILED,
    QUANTIZATION_FAILED,
};

/// @brief Converts ONNX models into NCNN param/bin files
class Onnx2NcnnConverter
{
public:
    Onnx2NcnnConverter();

    /// @brief Converts a model according to @a config
    /// @param config Conversion settings
    /// @return Conversion status
    /// @throws Throws exception if output metadata cannot be written
    ModelConversionStatus convert(const ModelConversionConfig& config);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo::tools
