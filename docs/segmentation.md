## 1- Why ARM-VO requires semantic segmentation?
ARM-VO benefits from semantic segmentation of the scene in 2 ways:

### 1.1 Scale Estimation
Scale estimation is done by:
- Fitting a plane to road regions
- Computing the camera height above the fitted plane (let's call it H)
- Dividing the actual camera height above ground (provided by user) to H

Estimating the road plane is the real challenge here. ARM-VO 1 used some heuristics to find the road plane via the triangulated 3D points. But the relative position of frames in a forward-looking camera is not ideal for triangulation and the ill-posed geometry often induces large errors in the triangulated points. ARM-VO 2 ditches the 3D space and estimates the road plane in 2D (image space). To do so, it performs homography matrix estimation using road keypoints. A semantic segmentation algorithm that is able to identify road pixels is used to filter-out non-road keypoints.

### 1.2 Dynamic Scenes
It's obvious that dynamic scenes can fool the motion estimation algorithm. Even RANSAC can fail if more than half of the keypoints belong to moving objects. ARM-VO uses the segmentation map to favor static keypoints over possibly-dynamic points. Any keypoint detected on road, sidewalks, walls, buildings, etc are considered static. Keypoints detected on cars, bikes, pedestrians, etc are considered possibly-dynamic.

## 2- What Segmentation algorithm is used?
ARM-VO uses BisenetV2 which is a popular model known for its runtime efficiency. Fortunately, I could find pre-trained weights on cityscapes dataset [link](https://github.com/CoinCheung/BiSeNet) and results on KITTI are good-enough. Note that Homography and Fundamental matrix estimations are robust to a level of outliers thanks to using RANSAC. Therefore, a rough segmentation map would be enough to classify keypoints into 1) road vs non-road, and 2) static vs non-static. I went a step further and downscaled the model from 512x1024 to 320x640 which reduced the runtime significantly while still producing good-enough results.

## 3- How to improve accuracy?
A higher-resolution network can improve segmentation accuracy, which may lead to better scale estimation and more robustness in dynamic scenes. BisenetV2 is a fully convolutional network and it is easy to export at the desired resolution as long as the dimensions are divisible by 32.
The following steps show how to use a model that takes inputs of size 416x832:

```bash
pip install onnxruntime
cd build/cli
./convert_model --model=path/to/model.onnx --height=416 --width=832
```

and then rebuild ARM-VO.
