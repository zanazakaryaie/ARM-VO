import sys

import cv2
import armvo


if len(sys.argv) < 3:
    sys.exit("Usage: python3 example.py config.yaml frame1.png [frame2.png ...]")

vo = armvo.ArmVo(armvo.ArmVoConfig.load(sys.argv[1]))
for path in sys.argv[2:]:
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read {path}")
    status, pose = vo.update(frame) if vo.is_initialized() else vo.initialize(frame)
    if pose is None:
        raise RuntimeError(f"Tracking failed: {status}")
    print(pose.rotation, pose.translation)
