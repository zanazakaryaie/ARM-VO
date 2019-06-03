# ARM-VO

**Authors**: [Zana Zakaryaie Nejad](http://imrid.net/) and [Ali Hosseininaveh](https://wp.kntu.ac.ir/hosseininaveh/Hosseininaveh_CV.html)

ARM-VO is an efficient monocular visual odometry algorithm designed for ARM processors. It uses NEON C intrinsics and multi-threading to accelerate keypoint detection and tracking. Check [this video](https://www.youtube.com/watch?v=2RwymYYxd5s&t=) to see the performance on Raspberry Pi 3 and Odroid XU4. 

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
# Notes
- If you get low FPS, check your power adapter. Raspberry Pi 3 runs ARM-VO at 8 frames per second (averagelly) if powered up with a 5V-2A adapter. 

- ARM-VO is a part of six-wheel surveying robot project named [MOOR](https://github.com/hosseininaveh/Moor). 




