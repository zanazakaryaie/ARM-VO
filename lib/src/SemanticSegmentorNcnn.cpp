#include "SemanticSegmentorNcnn.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <dlfcn.h>
//#include <ncnn/net.h>
//#include <ncnn/cpu.h>
#include <net.h>
#include <cpu.h>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace armvo
{

class SemanticSegmentorNcnn::Impl
{
    ncnn::Net mNet;

    const float MEAN[3] = {0.3257f * 255.f,
                           0.3690f * 255.f,
                           0.3223f * 255.f};

    const float NORM[3] = {1.f / (0.2112f * 255.f),
                           1.f / (0.2148f * 255.f),
                           1.f / (0.2115f * 255.f)};

    const std::string MODEL_INPUT_NAME = "input_image";
    int modelInputWidth = 640;
    int modelInputHeight = 320;

    const std::string MODEL_OUTPUT_NAME = "preds";
    int modelOutputWidth = 640;
    int modelOutputHeight = 320;
    static const int MODEL_OUTPUT_CHANNELS = 19;

    cv::Mat mSegmentationMap;
    bool mRunsOnGPU = false;

public:

    Impl()
    {
#if NCNN_VULKAN
        mRunsOnGPU = ncnn::get_gpu_count() > 0;
#endif
        mNet.opt = createFastestOption();

        fs::path dir = modelDir();

        {
            std::ifstream shapeFilehandle((dir / "model.shape").string());
            if (!shapeFilehandle)
            {
                throw std::runtime_error("Failed to load shape info");
            }
            shapeFilehandle >> modelInputHeight >> modelInputWidth;
            modelOutputWidth = modelInputWidth;
            modelOutputHeight = modelInputHeight;
        }

        if (mNet.load_param((dir / "model.param").c_str()) || mNet.load_model((dir / "model.bin").c_str()))
        {
            throw std::runtime_error("Failed to load the segmentation model");
        }

        mSegmentationMap.create(modelOutputHeight, modelOutputWidth, CV_8UC1);
    }

    bool runsOnCoProcessor() const noexcept
    {
        return mRunsOnGPU;
    }

    cv::Mat segment(const cv::Mat& frame, PixelFormat pixelFormat)
    {
        if (!frame.isContinuous())
        {
            return cv::Mat();
        }

        const int type = getNcnnType(pixelFormat);
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, type, frame.cols, frame.rows, modelInputWidth, modelInputHeight);
        in.substract_mean_normalize(MEAN, NORM);

        auto ex = mNet.create_extractor();
        if (ex.input(MODEL_INPUT_NAME.c_str(), in) != 0)
        {
            return cv::Mat();
        }

        ncnn::Mat out;
        if (ex.extract(MODEL_OUTPUT_NAME.c_str(), out) != 0)
        {
            return cv::Mat();
        }

        // Argmax
        for (int y=0; y<modelOutputHeight; y++)
        {
            const float* rowPtrs[MODEL_OUTPUT_CHANNELS];
            for (int c=0; c<MODEL_OUTPUT_CHANNELS; c++)
            {
                rowPtrs[c] = out.channel(c).row(y);
            }
            uchar* map = mSegmentationMap.ptr<uchar>(y);

            for (int x=0; x<modelOutputWidth; x++)
            {
                int bestClass = 0;
                float bestProb = rowPtrs[0][x];
                for (int c=1; c<MODEL_OUTPUT_CHANNELS; c++)
                {
                    const float prob = rowPtrs[c][x];
                    if (prob > bestProb)
                    {
                        bestProb = prob;
                        bestClass = c;
                    }
                }
                map[x] = bestClass;
            }
        }

        cv::Mat output;
        cv::resize(mSegmentationMap, output, frame.size(), 0, 0, cv::INTER_NEAREST);

        return output;
    }

    static cv::Mat getRoadMask(const cv::Mat& segmentationMap)
    {
        constexpr int ROAD_LABEL = 0;
        return segmentationMap == ROAD_LABEL;
    }

    static cv::Mat getStaticMask(const cv::Mat& segmentationMap)
    {
        cv::Mat mask(segmentationMap.size(), CV_8UC1);
        const uint8_t* const src = segmentationMap.ptr<uint8_t>();
        uint8_t* dst = mask.ptr<uint8_t>();
        for (uint32_t i=0; i<mask.total(); i++)
        {
            dst[i] = (src[i] <= 10 ? 255 : 0);
        }
        return mask;
    }

private:

    static fs::path modelDir()
    {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&modelDir), &info) == 0)
        {
            throw std::runtime_error("dladdr failed: cannot locate shared object");
        }

        fs::path libPath = fs::canonical(info.dli_fname);

        return libPath.parent_path() / "armvo" / "model";
    }

    static int getNcnnType(PixelFormat pixelFormat)
    {
        int type;
        switch (pixelFormat)
        {
            case PixelFormat::Gray:
                type = ncnn::Mat::PIXEL_GRAY2RGB;
                break;
            case PixelFormat::BGR:
                type = ncnn::Mat::PIXEL_BGR2RGB;
                break;
            case PixelFormat::RGB:
                type = ncnn::Mat::PIXEL_RGB;
                break;
        }
        return type;
    }

    static ncnn::Option createFastestOption()
    {
        ncnn::Option opt;

        opt.lightmode = false;
        opt.num_threads = std::max(1u, std::thread::hardware_concurrency());
        opt.openmp_blocktime = 0;
        opt.flush_denormals  = 3;

        opt.use_packing_layout = true;
        opt.use_winograd_convolution = true;
        opt.use_sgemm_convolution = true;
        opt.use_winograd23_convolution = true;
        opt.use_winograd43_convolution = true;
        opt.use_winograd63_convolution = true;
        opt.use_int8_inference = true;
        opt.use_a53_a55_optimized_kernel = ncnn::is_current_thread_running_on_a53_a55();

        opt.use_bf16_storage = ncnn::cpu_support_arm_bf16() || ncnn::cpu_support_x86_avx512_bf16();

#if NCNN_VULKAN
        if (ncnn::get_gpu_count() > 0)
        {
            opt.use_vulkan_compute = true;

            opt.use_fp16_packed = true;
            opt.use_fp16_storage = true;
            opt.use_fp16_arithmetic = true;
            opt.use_int8_packed = true;
            opt.use_int8_storage = true;
            opt.use_int8_arithmetic = true;

            opt.use_shader_pack8 = true;
            opt.use_subgroup_ops = true;
            opt.use_shader_local_memory = true;
            opt.use_cooperative_matrix = true;
        }
#endif

        return opt;
    }
};

SemanticSegmentorNcnn::SemanticSegmentorNcnn()
{
    mImpl = std::make_shared<Impl>();
}

bool SemanticSegmentorNcnn::runsOnCoProcessor()
{
    return mImpl->runsOnCoProcessor();
}

cv::Mat SemanticSegmentorNcnn::segment(const cv::Mat& frame, PixelFormat pixelFormat)
{
    return mImpl->segment(frame, pixelFormat);
}

cv::Mat SemanticSegmentorNcnn::getRoadMask(const cv::Mat& segmentationMap)
{
    return Impl::getRoadMask(segmentationMap);
}

cv::Mat SemanticSegmentorNcnn::getStaticMask(const cv::Mat& segmentationMap)
{
    return Impl::getStaticMask(segmentationMap);
}

}
