#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armvo/ARM_VO.hpp>
#include <armvo/Types.hpp>

namespace py = pybind11;

namespace
{

py::array_t<float> matx33ToArray(const cv::Matx33f& matrix)
{
    auto array = py::array_t<float>({3, 3});
    std::memcpy(array.mutable_data(), matrix.val, 9 * sizeof(float));
    return array;
}

cv::Matx33f arrayToMatx33(const py::object& value)
{
    py::array_t<float, py::array::c_style | py::array::forcecast> array(value);
    const py::buffer_info info = array.request();

    if (info.ndim != 2 || info.shape[0] != 3 || info.shape[1] != 3)
    {
        throw std::invalid_argument("intrinsics and rotation must have shape (3, 3)");
    }

    cv::Matx33f matrix;
    std::memcpy(matrix.val, info.ptr, 9 * sizeof(float));
    return matrix;
}

py::array_t<float> matx31ToArray(const cv::Matx31f& vector)
{
    auto array = py::array_t<float>({3});
    std::memcpy(array.mutable_data(), vector.val, 3 * sizeof(float));
    return array;
}

cv::Matx31f arrayToMatx31(const py::object& value)
{
    py::array_t<float, py::array::c_style | py::array::forcecast> array(value);
    const py::buffer_info info = array.request();

    const bool isFlatVector = info.ndim == 1 && info.shape[0] == 3;
    const bool isColumnVector = info.ndim == 2 && info.shape[0] == 3 && info.shape[1] == 1;
    if (!isFlatVector && !isColumnVector)
    {
        throw std::invalid_argument("translation must have shape (3,) or (3, 1)");
    }

    cv::Matx31f vector;
    std::memcpy(vector.val, info.ptr, 3 * sizeof(float));
    return vector;
}

py::object distortionsToObject(const armvo::CameraConfig& config)
{
    if (!config.distortions.has_value())
    {
        return py::none();
    }

    cv::Mat distortions;
    config.distortions.value().reshape(1, 1).convertTo(distortions, CV_64F);

    auto array = py::array_t<double>({static_cast<py::ssize_t>(distortions.total())});
    std::memcpy(array.mutable_data(), distortions.ptr<double>(), distortions.total() * sizeof(double));
    return array;
}

void setDistortions(armvo::CameraConfig& config, const py::object& value)
{
    if (value.is_none())
    {
        config.distortions.reset();
        return;
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> array(value);
    const py::buffer_info info = array.request();

    const bool isVector = info.ndim == 1;
    const bool isRowOrColumnVector = info.ndim == 2 && (info.shape[0] == 1 || info.shape[1] == 1);
    if (!isVector && !isRowOrColumnVector)
    {
        throw std::invalid_argument("distortions must be a 1-D array, a row vector, a column vector, or None");
    }

    const auto count = static_cast<int>(array.size());
    cv::Mat distortions(1, count, CV_64F);
    std::memcpy(distortions.ptr<double>(), info.ptr, static_cast<std::size_t>(count) * sizeof(double));
    config.distortions = distortions;
}

void requireIntRange(int value, int minValue, int maxValue, const char* name)
{
    if (value < minValue || value > maxValue)
    {
        throw std::out_of_range(std::string(name) + " must be in [" + std::to_string(minValue) + ", " +
                                std::to_string(maxValue) + "]");
    }
}

cv::Mat frameToMat(const py::array& frame)
{
    if (!py::isinstance<py::array_t<uint8_t>>(frame))
    {
        throw std::invalid_argument("frame must be a uint8 NumPy array");
    }

    if ((frame.flags() & py::array::c_style) == 0)
    {
        throw std::invalid_argument("frame must be C-contiguous");
    }

    const py::buffer_info info = frame.request();
    if (info.ndim == 2)
    {
        return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC1, info.ptr);
    }

    if (info.ndim == 3 && info.shape[2] == 3)
    {
        return cv::Mat(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC3, info.ptr);
    }

    throw std::invalid_argument("frame must have shape (height, width) or (height, width, 3)");
}

armvo::CameraConfig defaultCameraConfig()
{
    armvo::CameraConfig config;
    config.intrinsics = cv::Matx33f::eye();
    return config;
}

armvo::Pose identityPose()
{
    armvo::Pose pose;
    pose.rotation = cv::Matx33f::eye();
    pose.translation = cv::Matx31f::zeros();
    return pose;
}

armvo::ArmVoConfig defaultArmVoConfig()
{
    armvo::ArmVoConfig config;
    config.camera = defaultCameraConfig();
    return config;
}

py::tuple runInitialize(armvo::ArmVo& vo, const py::array& frame)
{
    armvo::Pose pose;
    const cv::Mat mat = frameToMat(frame);
    armvo::Status status;
    {
        py::gil_scoped_release release;
        status = vo.initialize(mat, pose);
    }
    if (status != armvo::Status::SUCCESS)
    {
        return py::make_tuple(status, py::none());
    }
    return py::make_tuple(status, pose);
}

py::tuple runUpdate(armvo::ArmVo& vo, const py::array& frame)
{
    armvo::Pose pose;
    const cv::Mat mat = frameToMat(frame);
    armvo::Status status;
    {
        py::gil_scoped_release release;
        status = vo.update(mat, pose);
    }
    if (status != armvo::Status::SUCCESS && status != armvo::Status::FRAME_SKIPPED)
    {
        return py::make_tuple(status, py::none());
    }
    return py::make_tuple(status, pose);
}

} // namespace

PYBIND11_MODULE(armvo, module)
{
    module.doc() = "Python bindings for the ARM-VO core library.";

    py::enum_<armvo::PixelFormat>(module, "PixelFormat")
        .value("GRAY", armvo::PixelFormat::Gray)
        .value("BGR", armvo::PixelFormat::BGR)
        .value("RGB", armvo::PixelFormat::RGB);

    py::enum_<armvo::Status>(module, "Status")
        .value("SUCCESS", armvo::Status::SUCCESS)
        .value("INVALID_FRAME", armvo::Status::INVALID_FRAME)
        .value("NOT_INITIALIZED", armvo::Status::NOT_INITIALIZED)
        .value("NOT_ENOUGH_KEYPOINTS", armvo::Status::NOT_ENOUGH_KEYPOINTS)
        .value("TRACK_LOST", armvo::Status::TRACK_LOST)
        .value("FRAME_SKIPPED", armvo::Status::FRAME_SKIPPED)
        .value("SCALE_ESTIMATION_FAILURE", armvo::Status::SCALE_ESTIMATION_FAILURE);

    py::class_<armvo::CameraConfig>(module, "CameraConfig")
        .def(py::init(&defaultCameraConfig))
        .def_property(
            "intrinsics",
            [](const armvo::CameraConfig& config) { return matx33ToArray(config.intrinsics); },
            [](armvo::CameraConfig& config, const py::object& value) { config.intrinsics = arrayToMatx33(value); })
        .def_readwrite("pixel_format", &armvo::CameraConfig::pixelFormat)
        .def_readwrite("fps", &armvo::CameraConfig::fps)
        .def_readwrite("height", &armvo::CameraConfig::height)
        .def_property("distortions", &distortionsToObject, &setDistortions);

    py::class_<armvo::KeypointDetectorConfig>(module, "KeypointDetectorConfig")
        .def(py::init<>())
        .def_property(
            "max_number_of_points",
            [](const armvo::KeypointDetectorConfig& config) { return config.maxNumberOfPoints; },
            [](armvo::KeypointDetectorConfig& config, int value) {
                requireIntRange(value, 0, 65535, "max_number_of_points");
                config.maxNumberOfPoints = static_cast<uint16_t>(value);
            })
        .def_property(
            "response_threshold",
            [](const armvo::KeypointDetectorConfig& config) { return config.responseThreshold; },
            [](armvo::KeypointDetectorConfig& config, int value) {
                requireIntRange(value, 0, 255, "response_threshold");
                config.responseThreshold = static_cast<uint8_t>(value);
            })
        .def_property(
            "number_of_image_grid_rows",
            [](const armvo::KeypointDetectorConfig& config) { return config.numberOfImageGridRows; },
            [](armvo::KeypointDetectorConfig& config, int value) {
                requireIntRange(value, 0, 255, "number_of_image_grid_rows");
                config.numberOfImageGridRows = static_cast<uint8_t>(value);
            })
        .def_property(
            "number_of_image_grid_cols",
            [](const armvo::KeypointDetectorConfig& config) { return config.numberOfImageGridCols; },
            [](armvo::KeypointDetectorConfig& config, int value) {
                requireIntRange(value, 0, 255, "number_of_image_grid_cols");
                config.numberOfImageGridCols = static_cast<uint8_t>(value);
            });

    py::class_<armvo::KeypointTrackerConfig>(module, "KeypointTrackerConfig")
        .def(py::init<>())
        .def_property(
            "window_size",
            [](const armvo::KeypointTrackerConfig& config) { return config.windowSize; },
            [](armvo::KeypointTrackerConfig& config, int value) {
                requireIntRange(value, 0, 255, "window_size");
                config.windowSize = static_cast<uint8_t>(value);
            });

    py::class_<armvo::ArmVoConfig>(module, "ArmVoConfig")
        .def(py::init(&defaultArmVoConfig))
        .def_static("load", &armvo::ArmVoConfig::load, py::arg("filepath"))
        .def_readwrite("camera", &armvo::ArmVoConfig::camera)
        .def_readwrite("keypoint_detector", &armvo::ArmVoConfig::keypointDetector)
        .def_readwrite("keypoint_tracker", &armvo::ArmVoConfig::keypointTracker)
        .def_readwrite("max_vehicle_speed", &armvo::ArmVoConfig::maxVehicleSpeed);

    py::class_<armvo::Pose>(module, "Pose")
        .def(py::init(&identityPose))
        .def_property(
            "rotation",
            [](const armvo::Pose& pose) { return matx33ToArray(pose.rotation); },
            [](armvo::Pose& pose, const py::object& value) { pose.rotation = arrayToMatx33(value); })
        .def_property(
            "translation",
            [](const armvo::Pose& pose) { return matx31ToArray(pose.translation); },
            [](armvo::Pose& pose, const py::object& value) { pose.translation = arrayToMatx31(value); });

    py::class_<armvo::ArmVo>(module, "ArmVo")
        .def(py::init<const armvo::ArmVoConfig&>(), py::arg("config"))
        .def("is_initialized", &armvo::ArmVo::isInitialized)
        .def("initialize", &runInitialize, py::arg("frame"))
        .def("update", &runUpdate, py::arg("frame"));
}
