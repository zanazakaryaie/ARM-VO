# CI dependency images

`ci.yaml` uses native x86_64 and ARM64 runners. Each pulls
`ghcr.io/<owner>/<repo>/ci-deps:<arch>-<dependency-hash>`, or builds it when unavailable.
The hash covers the image recipe and dependency CMake files. Only successful
master runs publish images; PRs can build missing images without publishing them.

OpenCV, ncnn, and Catch2 are built through `3rd-party`. Ubuntu supplies the
compiler, Python, NumPy, pybind11, Protobuf, and image/GUI development libraries.