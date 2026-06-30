#include "Onnx2NcnnConverter.hpp"
#include "FileTools.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace armvo::tools
{

class Onnx2NcnnConverter::Impl
{
public:

    Impl() = default;

    ModelConversionStatus convert(const ModelConversionConfig& config)
    {
        if (!fs::is_regular_file(config.onnxFilepath) )
        {
            return ModelConversionStatus::INVALID_ONNX_FILEPATH;
        }

        if (config.quantization.has_value() &&
            !fs::is_directory(config.quantization.value().calibImagesDir))
        {
            return ModelConversionStatus::QUANTIZATION_FAILED;
        }

        fs::create_directories(config.outputDir);

        const std::string onnxFilename = fs::path(config.onnxFilepath).stem().string();
        const std::string fixedSizeOnnxFilepath = config.outputDir + "/" + onnxFilename + "_fixed.onnx";
        const std::string inputShapeStr = "1,3," + std::to_string(config.inputHeight) + "," + std::to_string(config.inputWidth);
        const std::string exportCmd = createCommand({"python3", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
                                                     "--input_name", "input_image",
                                                     "--input_shape", inputShapeStr,
                                                     config.onnxFilepath, fixedSizeOnnxFilepath});

        if (!runCommand(exportCmd))
        {
            return ModelConversionStatus::CONVERSION_FAILED;
        }

        {
            const std::string shapeFilepath = config.outputDir + "/" + "model.shape";
            std::ofstream shapeFilehandle(shapeFilepath);
            if (!shapeFilehandle.is_open())
            {
                throw std::runtime_error("Failed to store model shape file");
            }
            shapeFilehandle << config.inputHeight << " " << config.inputWidth << "\n";
        }

        const std::string paramFilepath = config.outputDir + "/" + "model.param";
        const std::string binFilepath = config.outputDir + "/" + "model.bin";
        const std::string convCmd = createCommand({"onnx2ncnn", fixedSizeOnnxFilepath, paramFilepath, binFilepath});

        bool isConversionDone = runCommand(convCmd);

        fs::remove(fixedSizeOnnxFilepath);

        if (!isConversionDone)
        {
            return ModelConversionStatus::CONVERSION_FAILED;
        }

        if (config.optimize)
        {
            const std::string optCmd = createCommand({"ncnnoptimize", paramFilepath, binFilepath, paramFilepath, binFilepath, "0"});
            if (!runCommand(optCmd))
            {
                return ModelConversionStatus::OPTIMIZATION_FAILED;
            }
        }

        if (config.quantization.has_value())
        {
            const auto quantizationConfig = config.quantization.value();

            const std::string imageListFilepath = generateImageListFile(quantizationConfig.calibImagesDir, config.outputDir);
            if (imageListFilepath.empty())
            {
                return ModelConversionStatus::QUANTIZATION_FAILED;
            }
            const std::string tableFilepath = config.outputDir + "/model.table";
            const std::string meanStr = "mean=" + convertArrayToString(quantizationConfig.mean);
            const std::string normStr = "norm=" + convertArrayToString(quantizationConfig.norm);
            const std::string shapeStr = "shape=" + convertArrayToString(quantizationConfig.modelInputShape);
            const std::string pixelStr = getPixelStr(quantizationConfig.modelInputPixelFormat);
            const uint32_t numThreads = std::max(1u, std::thread::hardware_concurrency()/2);
            const std::string threadsStr = "thread=" + std::to_string(numThreads);
            const std::string quantMethod = "method=kl";
            const std::string type = "type=0";

            const std::string quantTableCmd = createCommand({"ncnn2table", paramFilepath, binFilepath,
                                                              imageListFilepath, tableFilepath,
                                                              meanStr, normStr, shapeStr, pixelStr,
                                                              threadsStr, quantMethod, type});
            if (!runCommand(quantTableCmd))
            {
                return ModelConversionStatus::QUANTIZATION_FAILED;
            }

            const std::string quantCmd = createCommand({"ncnn2int8", paramFilepath, binFilepath, paramFilepath, binFilepath, tableFilepath});
            if (!runCommand(quantCmd))
            {
                return ModelConversionStatus::QUANTIZATION_FAILED;
            }
        }

        return ModelConversionStatus::SUCCESS;
    }

private:

    static std::string createCommand(const std::vector<std::string>& tokens)
    {
        std::string command;
        for (size_t i = 0; i < tokens.size(); i++)
        {
            if (i > 0)
            {
                command += " ";
            }
            command += shellQuote(tokens[i]);
        }
        return command;
    }

    static std::string shellQuote(const std::string& token)
    {
        std::string quoted = "'";
        for (const char c : token)
        {
            if (c == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted += c;
            }
        }
        quoted += "'";
        return quoted;
    }

    static std::string generateImageListFile(const std::string& calibDataDir, const std::string& outDir)
    {
        std::string txtFilepath = outDir + "/imageList.txt";
        std::ofstream file(txtFilepath);
        if (!file.is_open())
        {
            return {};
        }
        const auto imageFilepaths = FileTools::listFramesInDir(calibDataDir);
        if (imageFilepaths.empty())
        {
            return {};
        }
        for (const auto& imageFilepath : imageFilepaths)
        {
            file << imageFilepath << "\n";
        }
        file.close();
        return txtFilepath;
    }

    template<typename T, std::size_t N>
    static std::string convertArrayToString(const std::array<T, N>& array)
    {
        std::string output = "[";
        for (int i=0; i<N-1; i++)
        {
            output += std::to_string(array[i]) + ",";
        }
        output += std::to_string(array.back()) + "]";
        return output;
    }

    static std::string getPixelStr(armvo::PixelFormat pixelFormat)
    {
        switch(pixelFormat)
        {
            case armvo::PixelFormat::Gray:
                return "pixel=GRAY";
            case armvo::PixelFormat::BGR:
                return "pixel=BGR";
            case armvo::PixelFormat::RGB:
                return "pixel=RGB";
            default:
                throw std::logic_error("Unsupported pixel format");
        }
    }

    static bool runCommand(const std::string& command)
    {
        return std::system(command.c_str()) == 0;
    }
};

Onnx2NcnnConverter::Onnx2NcnnConverter()
{
    mImpl = std::make_shared<Impl>();
}

ModelConversionStatus Onnx2NcnnConverter::convert(const ModelConversionConfig& config)
{
    return mImpl->convert(config);
}

}
