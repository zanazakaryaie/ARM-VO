# ARM-VO

**Authors**: [Zana Zakaryaie Nejad](http://imrid.net/) and Ali Hosseininaveh

ARM-VO is an efficient monocular visual odometry algorithm designed for ARM processors. It uses NEON C intrinsics and multi-threading to accelerate keypoint detection and tracking. Results on KITTI dataset confirms 8 FPS performance on Raspberry Pi 3. 

# Dependencies
- Cmake
- OpenCV ([built with TBB](https://www.theimpossiblecode.com/blog/intel-tbb-on-raspberry-pi/))

# How to build?
```
git clone https://github.com/zanazakaryaie/ARM-VO.git
cd ARM-VO
cmake .
make
```
# Test on KITTI dataset
Download the odometry dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Open a terminal and type:
```
./ARM_VO pathToData paramsFileName
```



