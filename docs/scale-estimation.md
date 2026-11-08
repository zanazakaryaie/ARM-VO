# Scale estimation

Monocular vision can recover the direction of camera motion, but not its
distance in meters. ARM-VO uses the known camera height above the road to
recover this scale, assuming the visible road is approximately flat.

Road keypoints are tracked between two frames, both forward and backward.
Each direction provides a homography: an image transformation describing how
the road appears to move. Combining it with the estimated camera rotation,
translation direction, and known camera height gives a distance estimate.
This works directly with image points, without reconstructing a 3D road plane.

Estimates that exceed the maximum expected vehicle speed or change too sharply
from the previous scale are rejected. The remaining estimates are combined
using a weighted geometric mean, giving more influence to homographies that
fit their tracked points better. If only one estimate is accepted, it is used
directly.

When neither direction provides an accepted estimate, the previous scale is
reused. If no scale has been accepted yet, scale estimation fails.
