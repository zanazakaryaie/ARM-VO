#include "SemanticSegmentorTensorRT.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace armvo
{
namespace
{

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
};

struct CudaStreamDeleter
{
    void operator()(cudaStream_t stream) const noexcept
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }
};

using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine>;
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext>;
using DevicePtr = std::unique_ptr<void, CudaDeleter>;
using StreamPtr = std::unique_ptr<std::remove_pointer<cudaStream_t>::type, CudaStreamDeleter>;

constexpr std::array<float, 3> kMean{0.3257f * 255.f,
                                     0.3690f * 255.f,
                                     0.3223f * 255.f};

constexpr std::array<float, 3> kNorm{1.f / (0.2112f * 255.f),
                                     1.f / (0.2148f * 255.f),
                                     1.f / (0.2115f * 255.f)};

constexpr const char* kModelInputName = "input_image";
constexpr const char* kModelOutputName = "preds";
constexpr int kModelOutputChannels = 19;

bool hasDynamicDim(const nvinfer1::Dims& dims)
{
    for (int i = 0; i < dims.nbDims; i++)
    {
        if (dims.d[i] < 0)
        {
            return true;
        }
    }
    return false;
}

size_t volume(const nvinfer1::Dims& dims)
{
    size_t result = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        if (dims.d[i] < 0)
        {
            throw std::runtime_error("TensorRT binding dimensions are not fully specified");
        }
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
}

void checkCuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

class SemanticSegmentorTensorRT::Impl
{
    Logger mLogger;
    RuntimePtr mRuntime;
    EnginePtr mEngine;
    ContextPtr mContext;
    DevicePtr mDeviceInput;
    DevicePtr mDeviceOutput;
    StreamPtr mStream;
    std::vector<float> mInput;
    std::vector<float> mOutput;
    cv::Mat mSegmentationMap;

    int mModelInputWidth = 640;
    int mModelInputHeight = 320;
    int mModelOutputWidth = 640;
    int mModelOutputHeight = 320;

public:

    Impl()
    {
        initLibNvInferPlugins(&mLogger, "");

        const fs::path dir = modelDir();
        loadShape(dir / "model.shape");

        mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
        if (mRuntime == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        const std::vector<char> engineData = readFile(dir / "model.engine");
        mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (mEngine == nullptr)
        {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }

        mContext.reset(mEngine->createExecutionContext());
        if (mContext == nullptr)
        {
            throw std::runtime_error("Failed to create TensorRT execution context");
        }

        validateTensorMetadata();

        nvinfer1::Dims inputDims = mEngine->getTensorShape(kModelInputName);
        if (hasDynamicDim(inputDims))
        {
            inputDims = fixedInputDims(inputDims);
            if (!mContext->setInputShape(kModelInputName, inputDims))
            {
                throw std::runtime_error("Failed to set TensorRT input dimensions");
            }
        }

        const nvinfer1::Dims outputDims = mContext->getTensorShape(kModelOutputName);
        updateOutputShape(outputDims);

        mInput.resize(static_cast<size_t>(3 * mModelInputHeight * mModelInputWidth));
        mOutput.resize(volume(outputDims));
        mSegmentationMap.create(mModelOutputHeight, mModelOutputWidth, CV_8UC1);

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;
        checkCuda(cudaMalloc(&deviceInput, mInput.size() * sizeof(float)), "Failed to allocate TensorRT input buffer");
        checkCuda(cudaMalloc(&deviceOutput, mOutput.size() * sizeof(float)), "Failed to allocate TensorRT output buffer");
        mDeviceInput.reset(deviceInput);
        mDeviceOutput.reset(deviceOutput);

        cudaStream_t rawStream = nullptr;
        checkCuda(cudaStreamCreate(&rawStream), "Failed to create TensorRT CUDA stream");
        mStream.reset(rawStream);

        if (!mContext->setTensorAddress(kModelInputName, mDeviceInput.get()) ||
            !mContext->setTensorAddress(kModelOutputName, mDeviceOutput.get()))
        {
            throw std::runtime_error("Failed to bind TensorRT device buffers");
        }
    }

    bool runsOnCoProcessor() const noexcept
    {
        return true;
    }

    cv::Mat segment(const cv::Mat& frame, PixelFormat pixelFormat)
    {
        if (!frame.isContinuous())
        {
            return cv::Mat();
        }

        preprocess(frame, pixelFormat);

        if (cudaMemcpyAsync(mDeviceInput.get(), mInput.data(), mInput.size() * sizeof(float),
                            cudaMemcpyHostToDevice, mStream.get()) != cudaSuccess)
        {
            return cv::Mat();
        }

        if (!mContext->enqueueV3(mStream.get()))
        {
            return cv::Mat();
        }

        if (cudaMemcpyAsync(mOutput.data(), mDeviceOutput.get(), mOutput.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, mStream.get()) != cudaSuccess)
        {
            return cv::Mat();
        }

        if (cudaStreamSynchronize(mStream.get()) != cudaSuccess)
        {
            return cv::Mat();
        }

        argmax();

        cv::Mat output;
        cv::resize(mSegmentationMap, output, frame.size(), 0, 0, cv::INTER_NEAREST);
        return output;
    }

private:

    static fs::path modelDir()
    {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&modelDir), &info) == 0)
        {
            throw std::runtime_error("dladdr failed: cannot locate shared object");
        }

        const fs::path libPath = fs::canonical(info.dli_fname);
        return libPath.parent_path() / "armvo" / "model";
    }

    void loadShape(const fs::path& shapeFilepath)
    {
        std::ifstream shapeFilehandle(shapeFilepath.string());
        if (!shapeFilehandle)
        {
            throw std::runtime_error("Failed to load TensorRT model shape info");
        }

        shapeFilehandle >> mModelInputHeight >> mModelInputWidth;
        mModelOutputHeight = mModelInputHeight;
        mModelOutputWidth = mModelInputWidth;
    }

    static std::vector<char> readFile(const fs::path& filepath)
    {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Failed to load TensorRT engine");
        }

        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> data(static_cast<size_t>(size));
        if (!file.read(data.data(), size))
        {
            throw std::runtime_error("Failed to read TensorRT engine");
        }
        return data;
    }

    void validateTensorMetadata() const
    {
        bool foundInput = false;
        bool foundOutput = false;

        for (int32_t i = 0; i < mEngine->getNbIOTensors(); i++)
        {
            const char* name = mEngine->getIOTensorName(i);
            if (name == nullptr)
            {
                continue;
            }

            const std::string tensorName(name);
            if (tensorName == kModelInputName)
            {
                foundInput = true;
                if (mEngine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
                {
                    throw std::runtime_error("TensorRT input tensor has unexpected IO mode");
                }
            }
            else if (tensorName == kModelOutputName)
            {
                foundOutput = true;
                if (mEngine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT)
                {
                    throw std::runtime_error("TensorRT output tensor has unexpected IO mode");
                }
            }
        }

        if (!foundInput || !foundOutput)
        {
            throw std::runtime_error("TensorRT engine does not contain expected input/output tensors");
        }

        if (mEngine->getTensorDataType(kModelInputName) != nvinfer1::DataType::kFLOAT ||
            mEngine->getTensorDataType(kModelOutputName) != nvinfer1::DataType::kFLOAT)
        {
            throw std::runtime_error("TensorRT segmentor currently expects FP32 input and output tensors");
        }
    }

    nvinfer1::Dims fixedInputDims(nvinfer1::Dims dims) const
    {
        if (dims.nbDims == 4)
        {
            dims.d[0] = 1;
            dims.d[1] = 3;
            dims.d[2] = mModelInputHeight;
            dims.d[3] = mModelInputWidth;
        }
        else if (dims.nbDims == 3)
        {
            dims.d[0] = 3;
            dims.d[1] = mModelInputHeight;
            dims.d[2] = mModelInputWidth;
        }
        else
        {
            throw std::runtime_error("TensorRT input binding has unsupported dimensions");
        }
        return dims;
    }

    void updateOutputShape(const nvinfer1::Dims& dims)
    {
        if (dims.nbDims < 3)
        {
            throw std::runtime_error("TensorRT output binding has unsupported dimensions");
        }

        if (dims.nbDims == 4 && dims.d[0] != 1)
        {
            throw std::runtime_error("TensorRT output binding has unsupported batch size");
        }

        const int channels = dims.d[dims.nbDims - 3];
        mModelOutputHeight = dims.d[dims.nbDims - 2];
        mModelOutputWidth = dims.d[dims.nbDims - 1];

        if (channels != kModelOutputChannels ||
            mModelOutputHeight != mModelInputHeight ||
            mModelOutputWidth != mModelInputWidth)
        {
            throw std::runtime_error("TensorRT output binding has unexpected shape");
        }
    }

    void preprocess(const cv::Mat& frame, PixelFormat pixelFormat)
    {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(mModelInputWidth, mModelInputHeight), 0, 0, cv::INTER_LINEAR);

        float* rPlane = mInput.data();
        float* gPlane = rPlane + mModelInputHeight * mModelInputWidth;
        float* bPlane = gPlane + mModelInputHeight * mModelInputWidth;

        for (int y = 0; y < mModelInputHeight; y++)
        {
            const uint8_t* row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < mModelInputWidth; x++)
            {
                const size_t index = static_cast<size_t>(y * mModelInputWidth + x);

                uint8_t r = 0;
                uint8_t g = 0;
                uint8_t b = 0;
                switch (pixelFormat)
                {
                    case PixelFormat::Gray:
                    {
                        r = g = b = row[x];
                        break;
                    }
                    case PixelFormat::BGR:
                    {
                        const cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
                        b = pixel[0];
                        g = pixel[1];
                        r = pixel[2];
                        break;
                    }
                    case PixelFormat::RGB:
                    {
                        const cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
                        r = pixel[0];
                        g = pixel[1];
                        b = pixel[2];
                        break;
                    }
                }

                rPlane[index] = (static_cast<float>(r) - kMean[0]) * kNorm[0];
                gPlane[index] = (static_cast<float>(g) - kMean[1]) * kNorm[1];
                bPlane[index] = (static_cast<float>(b) - kMean[2]) * kNorm[2];
            }
        }
    }

    void argmax()
    {
        const size_t channelStride = static_cast<size_t>(mModelOutputHeight * mModelOutputWidth);
        for (int y = 0; y < mModelOutputHeight; y++)
        {
            uint8_t* map = mSegmentationMap.ptr<uint8_t>(y);
            for (int x = 0; x < mModelOutputWidth; x++)
            {
                const size_t pixelIndex = static_cast<size_t>(y * mModelOutputWidth + x);
                int bestClass = 0;
                float bestProb = mOutput[pixelIndex];
                for (int c = 1; c < kModelOutputChannels; c++)
                {
                    const float prob = mOutput[static_cast<size_t>(c) * channelStride + pixelIndex];
                    if (prob > bestProb)
                    {
                        bestProb = prob;
                        bestClass = c;
                    }
                }
                map[x] = static_cast<uint8_t>(bestClass);
            }
        }
    }
};

SemanticSegmentorTensorRT::SemanticSegmentorTensorRT()
{
    mImpl = std::make_shared<Impl>();
}

bool SemanticSegmentorTensorRT::runsOnCoProcessor()
{
    return mImpl->runsOnCoProcessor();
}

cv::Mat SemanticSegmentorTensorRT::segment(const cv::Mat& frame, PixelFormat pixelFormat)
{
    return mImpl->segment(frame, pixelFormat);
}

} // namespace armvo
