# Python example

This example loads a sequence of images in order and prints the camera pose for each frame.

## Build and run

Install the [ARM-VO build dependencies](../../README.md#build-and-install), then install the Python dependencies:

```bash
sudo apt install python3-dev pybind11-dev python3-numpy python3-opencv
```

From the ARM-VO repository root, build the Python bindings:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON ..
make -j$(nproc)
sudo make install
cd ..
export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"
```

Run the example from the repository root in the same shell:

```bash
cd examples/python
python3 example.py config.yaml /path/to/sequences/00/image_2/*.png
```

Pass frames in chronological order. The wildcard above works with KITTI's zero-padded filenames.

## Configuration

The included [config.yaml](config.yaml) uses calibration values for KITTI sequences 00–02 with rectified color images. Set the camera intrinsics, height (meters), frame rate, and distortion coefficients for your own camera. Keep `pixel_format: "bgr"` for this example because OpenCV loads color images in BGR order. For unrectified images, add `distortions: [k1, k2, p1, p2, k3]` under `Camera` using your calibration values. See the other [KITTI configurations](../../cli/KITTI_configs) for examples.
