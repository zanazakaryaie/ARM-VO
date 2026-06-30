# KITTI Raw Unrectified Configs

These configs are for KITTI raw `*_extract.zip` archives, using the left color camera:

```text
<date>/<drive>_extract/image_02/data
```

Use `*_extract.zip` for unrectified images. Do not use `*_sync.zip` for this check, because KITTI describes the synced archive as rectified and undistorted.

The config values come from each date's official `calib_cam_to_cam.txt`:

- `K_02` -> `fx`, `fy`, `cx`, `cy`
- `D_02` -> `Camera.distortions`

## Download Links

Calibration files:

```text
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip
```

Recommended raw unrectified drive archives for quick checks:

```text
# Similar calibration/date to odometry seq 00
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_extract.zip

# Smaller 2011_09_26 drive
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0064/2011_09_26_drive_0064_extract.zip

# 2011_09_30 drive
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_extract.zip
```

Other odometry-training raw drive candidates:

```text
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0034/2011_10_03_drive_0034_extract.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0042/2011_10_03_drive_0042_extract.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0028/2011_09_30_drive_0028_extract.zip
```

## Run Example

```bash
./build/cli/run_armvo \
  --image_folder /path/to/2011_10_03/2011_10_03_drive_0027_extract/image_02/data \
  --config cli/KITTI_configs/unrectified/2011_10_03.yaml \
  --display false
```

Use the config file matching the date in the extracted archive path.
