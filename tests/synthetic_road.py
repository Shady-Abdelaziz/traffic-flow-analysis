"""The synthetic road the unit tests measure on, and the constants describing it.

Five test modules built the same identity-homography ``RoadFrame`` independently, each
with its own copy of the lane width and zone length. The numbers agreed, but nothing
held them together: changing the zone length in one file to test a boundary would have
left the others asserting against a road that no longer matched their own constants.

One pixel is one metre here, deliberately. It keeps the modules that use this frame --
speed fitting, counting, density, lane logic -- about their own arithmetic rather than
about the projection, which is measured by calibration and tested in ``test_geometry.py``
against a realistic perspective camera and in ``test_speed_calibrated.py`` against the
real ``calibration.yaml``.
"""

from __future__ import annotations

from trafficflow.geometry import Homography, RoadFrame

#: Synthetic dimensions. The real site's width is measured by calibration, not assumed.
LANE_WIDTH_M = 3.0
LANE_COUNT = 5
ZONE_M = 150.0

#: Frame rate of the archive's clips, and of every synthetic track built against them.
FPS = 10.0

#: The unit square, mapped to itself: metres are pixels.
UNIT_SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def identity_road_frame(**overrides) -> RoadFrame:
    """A road frame whose homography is the identity, so metres == pixels.

    ``overrides`` are passed through to :class:`RoadFrame`, which is how the tests that
    need a near-end offset or a pixel shift get one without rebuilding the homography.
    """
    return RoadFrame(
        homography=Homography.from_correspondences(UNIT_SQUARE, UNIT_SQUARE),
        lane_width_m=LANE_WIDTH_M,
        lane_count=LANE_COUNT,
        measurement_zone_m=ZONE_M,
        **overrides,
    )


def lane_centre(lane: int) -> float:
    """Lateral position of the middle of ``lane``, counting from 1 at the left edge."""
    return (lane - 0.5) * LANE_WIDTH_M
