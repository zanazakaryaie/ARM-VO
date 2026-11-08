# Scale estimation

Scale estimation is implemented by internal classes in `lib/src`, separately
from the motion and geometry utilities in `Geometry.hpp`.

| Component | Responsibility |
| --- | --- |
| `IScaleEstimator` | Produces fresh measurements for a frame pair. |
| `RoadPlaneScaleEstimator` | Samples and tracks road points, fits forward/backward homographies, and derives scale from camera height. |
| `ScaleFusion` | Computes the weighted geometric mean of accepted measurements. |
| `TemporalScaleFilter` | Applies speed and temporal jump limits before fusion, then retains the resulting scale for fallback. |

`ArmVo` supplies keypoints and the full segmentation map through `ScaleEstimationInput`.
It initializes the previous observations, supplies current observations, and
calls `estimate()` with the recovered rotation and unit translation. The
estimator's observations advance with the keypoint tracker only after a
successful pose update.

## Measurements and weights

`ScaleMeasurement::value` is the metric multiplier for the unit translation.
Its `weight` expresses relative influence in logarithmic fusion; it is not a
calibrated probability or inverse scale variance. The road estimator preserves
the existing weight: `1 / (1e-6 + mean squared Sampson residual)`.

An estimator returns zero or more fresh measurements. Road estimation returns
the successful forward and backward observations separately. This preserves
the order of operations: reject each observation, then fuse the survivors.
A single survivor passes through unchanged. Future estimators can supply more
measurements through the same interface, but their weights must be made
comparable before combining sources.

## Preserved temporal behavior

`TemporalScaleFilter` rejects measurements above `(maxVehicleSpeed / 3.6) / fps`,
or outside `[0.4, 2.5]` times the previous accepted fused scale. It compares all
measurements with the same previous scale before updating that scale.

When no measurement survives, it returns the previous scale, or no value if
none has ever been accepted. Reuse has no expiry, and reinitializing frame
observations does not reset this history. These are the existing behaviors;
changing them or adding new validity checks belongs in a separate algorithm
change. Estimators themselves never return a historical scale as fresh evidence.
