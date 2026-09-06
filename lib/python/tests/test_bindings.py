import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import armvo


def make_config(pixel_format=armvo.PixelFormat.GRAY):
    config = armvo.ArmVoConfig()
    config.camera.intrinsics = np.array(
        [[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    config.camera.pixel_format = pixel_format
    config.camera.fps = 10.0
    config.camera.height = 1.6
    config.keypoint_detector.max_number_of_points = 500
    config.keypoint_detector.response_threshold = 10
    config.keypoint_detector.number_of_image_grid_rows = 6
    config.keypoint_detector.number_of_image_grid_cols = 12
    config.keypoint_tracker.window_size = 21
    config.max_vehicle_speed = 90.0
    return config


class TestBindings(unittest.TestCase):
    def test_default_config_and_pose_have_numpy_arrays(self):
        config = armvo.ArmVoConfig()
        np.testing.assert_allclose(config.camera.intrinsics, np.eye(3, dtype=np.float32))
        self.assertEqual(config.camera.pixel_format, armvo.PixelFormat.GRAY)
        self.assertIsNone(config.camera.distortions)

        config.camera.distortions = [0.1, -0.2, 0.003, 0.004, 0.05]
        np.testing.assert_allclose(config.camera.distortions, np.array([0.1, -0.2, 0.003, 0.004, 0.05]))
        config.camera.distortions = None
        self.assertIsNone(config.camera.distortions)

        pose = armvo.Pose()
        np.testing.assert_allclose(pose.rotation, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(pose.translation, np.zeros(3, dtype=np.float32))

    def test_load_config(self):
        config_text = textwrap.dedent(
            """\
            %YAML:1.0

            Vehicle:
              max_speed: 80

            Camera:
              fx: 718.856
              fy: 718.856
              cx: 607.1928
              cy: 185.2157
              fps: 15
              height: 1.6
              pixel_format: "bgr"
              distortions: [0.1, -0.2, 0.003, 0.004, 0.05]

            Keypoint_Detector:
              num_points: 300
              response_threshold: 12
              num_image_grid_rows: 8
              num_image_grid_cols: 10

            Keypoint_Tracker:
              window_size: 21
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(config_text)
            config = armvo.ArmVoConfig.load(str(path))

        self.assertEqual(config.camera.pixel_format, armvo.PixelFormat.BGR)
        self.assertEqual(config.keypoint_detector.max_number_of_points, 300)
        self.assertEqual(config.keypoint_detector.response_threshold, 12)
        self.assertEqual(config.keypoint_detector.number_of_image_grid_rows, 8)
        self.assertEqual(config.keypoint_detector.number_of_image_grid_cols, 10)
        self.assertEqual(config.keypoint_tracker.window_size, 21)
        self.assertEqual(config.max_vehicle_speed, 80)
        np.testing.assert_allclose(
            config.camera.intrinsics,
            np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(config.camera.distortions, np.array([0.1, -0.2, 0.003, 0.004, 0.05]))

    def test_arm_vo_preconditions(self):
        vo = armvo.ArmVo(make_config())
        self.assertFalse(vo.is_initialized())

        blank_frame = np.zeros((480, 640), dtype=np.uint8)
        status, pose = vo.update(blank_frame)
        self.assertEqual(status, armvo.Status.NOT_INITIALIZED)
        self.assertIsNone(pose)

        status, pose = vo.initialize(np.empty((0, 0), dtype=np.uint8))
        self.assertEqual(status, armvo.Status.INVALID_FRAME)
        self.assertIsNone(pose)
        self.assertFalse(vo.is_initialized())

        wrong_channels = np.zeros((480, 640, 3), dtype=np.uint8)
        status, pose = vo.initialize(wrong_channels)
        self.assertEqual(status, armvo.Status.INVALID_FRAME)
        self.assertIsNone(pose)
        self.assertFalse(vo.is_initialized())

        status, pose = vo.initialize(blank_frame)
        self.assertEqual(status, armvo.Status.NOT_ENOUGH_KEYPOINTS)
        self.assertIsNone(pose)
        self.assertFalse(vo.is_initialized())

    def test_invalid_grid_config_throws(self):
        config = make_config()
        config.keypoint_detector.number_of_image_grid_rows = 0
        with self.assertRaises(RuntimeError):
            armvo.ArmVo(config)


if __name__ == "__main__":
    unittest.main()
