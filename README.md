# ARM-VO

**Authors**: Zana Zakaryaie Nejad and Ali Hosseininaveh

ARM-VO is an efficient monocular visual odometry algorithm designed for ARM processors. It uses NEON C intrinsics and multi-threading to accelerate keypoint detection and tracking. Results on KITTI dataset confirms 8 FPS performance on Raspberry Pi 3. 

# Dependencies
- Cmake
- OpenCV (built with TBB)

# How to build?
Clone this repo and navigate to the folder. Open a terminal and type:
```
cmake .
make
```
# Test on KITTI dataset
Download the odometry dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Open a terminal and type:
```
./ARM_VO pathToData paramsFileName
```



