# ARM-VO

**Authors**: [Zana Zakaryaie Nejad](http://imrid.net/) and [Ali Hosseininaveh](https://wp.kntu.ac.ir/hosseininaveh/Hosseininaveh_CV.html) 

ARM-VO is an efficient monocular visual odometry algorithm designed for ARM processors. It uses NEON C intrinsics and multi-threading to accelerate keypoint detection and tracking. Check [this video](https://www.youtube.com/watch?v=2RwymYYxd5s&t=) to see the performance on Raspberry Pi 3 and Odroid XU4.

[![Demo ARM-VO alpha](https://j.gifs.com/OM0NWR.gif)](https://www.youtube.com/watch?v=2RwymYYxd5s&t=)


## Dependencies
- Cmake
- OpenCV ([built with TBB](http://imrid.net/?p=3917))
- ROS (Optional)

## How to build?
```
git clone https://github.com/zanazakaryaie/ARM-VO.git
cd ARM-VO
cmake .
make
```
## Test on KITTI dataset
Download the odometry dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Open a terminal and type:
```
./ARM_VO pathToData paramsFileName
```
## ROS (Optional)
The ros node subscribes to "kitti/image" topic and publishes "arm_vo/pose" which is of type geometry_msg::Pose. To use the ROS node:
Open a terminal and type:
```
cd ROS
catkin_make
source devel/setup.bash
rosrun ARM_VO ARM_VO paramsFileName
```
Now, open another terminal and type:
```
cd ROS
source devel/setup.bash
rosrun ARM_VO ImagePublisher pathToData
```

## Limitations
- ARM-VO recovers the scale if the camera height and pitch angle are provided. Thus, it is not applicable for drones or hand-held cameras.

- The algorithm detects small-inter frame translations and pure rotations using GRIC but it doesn't decompose the estimated homography matrix. Track is lost if the camera rotates too much without translation. 

## Notes
- If you get low FPS, check your power adapter. Raspberry Pi 3 runs ARM-VO at 8 frames per second (averagelly) if powered up with a 5V-2A adapter. 

- If you use ARM-VO in an academic work, please cite: <br />
```
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

- ARM-VO is a part of six-wheel surveying robot project named [MOOR](https://github.com/hosseininaveh/Moor).

- [Here](https://github.com/wuyuanmm/ARM_VO_pybind) is a Python wrapper for ARM-VO.  




