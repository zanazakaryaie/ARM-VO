# ARM-VO

ARM-VO is an efficient monocular visual odometry library for on-road vehicles. It recovers scale using a known, fixed camera height and visible road. NEON intrinsics and multithreading accelerate keypoint detection and tracking.

## Results on KITTI dataset

| Sequence 05 | Sequence 07 | Sequence 10 |
|:---:|:---:|:---:|
| <img src="docs/assets/Sequence5.png" width="100%"> | <img src="docs/assets/Sequence7.png" width="100%"> | <img src="docs/assets/Sequence10.png" width="100%"> |

## What's new in v2?

- Deterministic results, more accurate scale estimation, and no camera pitch angle requirement.
- Faster tracking through KLT pyramid reuse and more robust motion estimation in dynamic scenes.
- Support for color (BGR or RGB) and distorted images, a much cleaner API, Python bindings, x86 support, and CI.
- ROS node examples removed (will be added in a near future)

## Build and install

ARM-VO requires C++17, CMake 3.20+, OpenCV, and either ncnn or TensorRT 8.6.x (TensorRT is preferred over ncnn when available).

Install build dependencies on Ubuntu:

```bash
sudo apt install build-essential git cmake pkg-config libopencv-dev
```

For the ncnn backend:

```bash
git clone --recursive --depth 1 --branch 20241226 https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_THREADS=ON -DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF -DNCNN_ENABLE_LTO=ON
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..
```

Then build ARM-VO:

```bash
git clone https://github.com/zanazakaryaie/ARM-VO.git
cd ARM-VO
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ..
```

### Build Options

| Option | Default | Purpose | Extra dependencies |
|---|---|---|---|
| `BUILD_TOOLS` | `ON` | Visualization, evaluation, and model conversion utilities | None |
| `BUILD_CLI` | `ON` | Command-line tools; requires `BUILD_TOOLS=ON` | None |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Python API | `sudo apt install python3-dev pybind11-dev python3-numpy` |
| `BUILD_TESTS` | `OFF` | Unit tests | Catch2 v2:<br>`git clone --branch v2.13.10 --depth 1 https://github.com/catchorg/Catch2.git`<br>`cd Catch2`<br> `mkdir build && cd build`<br>`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=/usr/local -DCATCH_BUILD_STATIC_LIBRARY=ON -DCATCH_BUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_HELPERS=ON ..`<br>`make -j$(nproc)`<br>`sudo make install`<br>`sudo ldconfig && cd ../..` |

## Run on KITTI dataset

Download the [color odometry images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip). From the ARM-VO repository root, run with the matching [configuration](cli/KITTI_configs):

```bash
./build/cli/run_armvo --image_folder=/path/to/sequences/00/image_2 --config=cli/KITTI_configs/rectified/Seq00-02.yaml
```

To evaluate accuracy, download the [ground-truth poses](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip) and add `--gt_poses=path/to/poses/00.txt`.

## How to use ARM-VO in your project?

Check the [C++ example](examples/cpp) or [Python example](examples/python) to see how to use ARM-VO in your project.

## Limitations
- The camera height and road requirements make ARM-VO unsuitable for drones, handheld cameras, or off-road vehicles.
- Large rotations without translation can lose tracking.

## Notes
- ARM-VO leverages a low-resolution (320x640) BisenetV2 segmentation model to 1) estimate scale, and 2) perform better in dynamic scenes. You can increase or decrease the resolution to trade-off between accuracy and FPS. Check [here](docs/segmentation.md) to read more and go through the required steps.
- If you get low FPS on single-board computers (e.g. Raspberry Pi), check your power adapter.

## For Developers
### Repository Layout

```text
.
├── cli/             Command-line tools for running ARM-VO
├── cmake/           CMake scripts
├── docs/            Documentation and README assets
├── examples/        C++ and Python usage examples
├── lib/             Core ARM-VO implementation and Python bindings
├── model/           BiseNetv2 model
├── tools/           Utilities for visualization, evaluation, etc.
└── CMakeLists.txt   Main CMake build file
```

### Running Tests
If you build ARM-VO with `-DBUILD_TESTS=ON`, you can run tests from the repo root by:
```bash
 ctest --test-dir build --output-on-failure
```

## License and citation

ARM-VO is [MIT licensed](LICENSE). For academic use, please cite:

```bibtex
@article{nejad2019arm,
  title={ARM-VO: an efficient monocular visual odometry for ground vehicles on ARM CPUs},
  author={Nejad, Zana Zakaryaie and Ahmadabadian, Ali Hosseininaveh},
  journal={Machine Vision and Applications},
  volume={30},
  number={6},
  pages={1061--1070},
  year={2019},
  publisher={Springer}
}
```

## TODOs
- Increase test coverage
- Add ROS examples
- Add redundancy for scale estimation (e.g. object priors)
- Support Bazel
