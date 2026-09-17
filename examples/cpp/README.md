# C++ example

This example loads a sequence of images in order and prints the camera pose for each frame.

## Build and run

First, [build and install ARM-VO](../../README.md#build-and-install). Then, from the repository root:

```bash
cd examples/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./my_app ../config.yaml /path/to/sequences/00/image_2/*.png
```

Pass frames in chronological order. The wildcard above works with KITTI's zero-padded filenames.

## Configuration

The included [config.yaml](config.yaml) uses calibration values for KITTI sequences 00–02 with rectified color images. Set the camera intrinsics, height (meters), frame rate, and distortion coefficients for your own camera. Keep `pixel_format: "bgr"` for this example because OpenCV loads color images in BGR order. For unrectified images, add `distortions: [k1, k2, p1, p2, k3]` under `Camera` using your calibration values. See the other [KITTI configurations](../../cli/KITTI_configs) for examples.
