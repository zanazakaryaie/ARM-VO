## 1- Why ARM-VO requires semantic segmentation?
ARM-VO benefits from semantic segmentation of the scene in 2 ways:

### 1.1 Scale Estimation

Scale estimation is done by:

- Fitting a plane to road regions
- Computing the camera height above the fitted plane (let's call it H)
- Dividing the actual camera height above ground (provided by user) to H

Estimating the road plane is the real challenge here. ARM-VO 1 used some heuristics to find the road plane via the triangulated 3D points. But the relative position of frames in a forward-looking camera is not ideal for triangulation and the ill-posed geometry often induces large errors in the triangulated points. ARM-VO 2 ditches the 3D space and estimates the road plane in 2D (image space). To do so, it performs homography matrix estimation using road keypoints. A semantic segmentation algorithm that is able to identify road pixels is used to filter-out non-road keypoints.

See [scale estimation](scale-estimation.md) for further details.

### 1.2 Dynamic Scenes
It's obvious that dynamic scenes can fool the motion estimation algorithm. Even RANSAC can fail if more than half of the keypoints belong to moving objects. ARM-VO uses the segmentation map to favor static keypoints over possibly-dynamic points. Any keypoint detected on road, sidewalks, walls, buildings, etc are considered static. Keypoints detected on cars, bikes, pedestrians, etc are considered possibly-dynamic.

## 2- What Segmentation algorithm is used?
ARM-VO uses BisenetV2 which is a popular model known for its runtime efficiency. Fortunately, I could find pre-trained weights on cityscapes dataset [link](https://github.com/CoinCheung/BiSeNet) and results on KITTI are good-enough. Note that Homography and Fundamental matrix estimations are robust to a level of outliers thanks to using RANSAC. Therefore, a rough segmentation map would be enough to classify keypoints into 1) road vs non-road, and 2) static vs non-static. I went a step further and downscaled the model from 512x1024 to 320x640 which reduced the runtime significantly while still producing good-enough results.

## 3- How do I change the segmentation resolution?

Both backends default to 640×320 pixels and FP32. A higher resolution can improve segmentation accuracy at the cost of inference time. BiSeNetV2 requires the width and height to each be a positive multiple of 32.

For example, to build for 832×416 pixels with FP16 enabled, run these commands from the repository root.

```bash
mkdir -p build && cd build
cmake -DSEG_WIDTH=832 -DSEG_HEIGHT=416 -DSEG_PRECISION=fp16 ..
make -j$(nproc)
```

To change the settings later, rerun `cmake` and `make` in the same build directory. Use `fp32` or `fp16` for precision. FP16 enables half precision where the hardware supports it. The output mask matches the original camera frame size at either precision.
