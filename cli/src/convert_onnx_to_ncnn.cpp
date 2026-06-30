#include <iostream>
#include <filesystem>
#include <cstdint>
#include <optional>
#include <string>

#include "cxxopts.hpp"

#include <armvo/tools/Onnx2NcnnConverter.hpp>

namespace fs = std::filesystem;

class CLI
{
public:

    static int run(int argc, char* argv[])
    {
        cxxopts::Options options(argv[0], "Converts the onnx model to what NCNN can digest");

        options.add_options()
            ("m,model", "Path to the onnx file", cxxopts::value<std::string>())
            ("h,height", "Desired input height", cxxopts::value<uint32_t>()->default_value("320"))
            ("w,width", "Desired input width", cxxopts::value<uint32_t>()->default_value("640"))
            ("c,calib-data-dir", "Path to the directory that stores calibration images (optional)", cxxopts::value<std::string>())
            ("p,pixel-format", "Pixel format of the model input (only needed if calibration data is provided)", cxxopts::value<std::string>())
            ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!result.count("model"))
        {
            std::cerr << "No model provided! Run with --help to see the required arguments" << std::endl;
            return -1;
        }

        armvo::tools::ModelConversionConfig config;
        config.onnxFilepath = fs::canonical(result["model"].as<std::string>());
        config.outputDir = fs::path(config.onnxFilepath).parent_path().string();
        config.inputHeight = result["height"].as<uint32_t>();
        config.inputWidth = result["width"].as<uint32_t>();

        if (result.count("calib-data-dir"))
        {
            if (!result.count("pixel-format"))
            {
                std::cerr << "--pixel-format is required when --calib-data-dir is provided" << std::endl;
                return -1;
            }

            armvo::tools::Int8QuantizationConfig quantConfig;
            quantConfig.calibImagesDir = fs::canonical(result["calib-data-dir"].as<std::string>());
            auto pixelFormatMaybe = string2PixelFormat(result["pixel-format"].as<std::string>());
            if (!pixelFormatMaybe.has_value())
            {
                return -1;
            }
            quantConfig.modelInputPixelFormat = pixelFormatMaybe.value();
            config.quantization = quantConfig;
        }

        armvo::tools::Onnx2NcnnConverter converter;
        const auto status = converter.convert(config);
        if (status == armvo::tools::ModelConversionStatus::SUCCESS)
        {
            return 0;
        }
        return -1;
    }

private:

    static std::optional<armvo::PixelFormat> string2PixelFormat(const std::string& str)
    {
        if (str == "rgb")
        {
            return armvo::PixelFormat::RGB;
        }
        else if (str == "bgr")
        {
            return armvo::PixelFormat::BGR;
        }
        else if (str == "gray")
        {
            return armvo::PixelFormat::Gray;
        }
        else
        {
            std::cerr << "Unsupported pixel format for model input" << std::endl;
            return std::nullopt;
        }
    }
};

int main(int argc, char* argv[])
{
    try
    {
        return CLI::run(argc, argv);
    }
    catch (std::exception& ex)
    {
        std::cerr << "Something went wrong" << std::endl;
        std::cerr << ex.what() << std::endl;
    }
    return -1;
}
