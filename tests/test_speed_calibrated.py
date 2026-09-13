"""Speed measurement against the *real* calibration, at the far end of the zone.

``test_speed.py`` mocks the road plane with an identity homography so one pixel is one
metre. That keeps those tests about the fitting, which is right, but it also means every
one of them runs at a scale this camera never has -- and the scale is where the errors
were. Under the real homography a pixel is about a quarter of a metre at the near edge
of the measurement zone and roughly 1.8 m at the far edge, a factor of seven across one
image, and a tolerance that ignores that is either loose near the camera or narrower
than a single pixel far from it.

These tests use ``calibration.yaml`` and ``config/site.yaml`` as they ship.
"""

from __future__ import annotations

import numpy as np
import pytest

from trafficflow.config import Paths, load_calibration, load_site
from trafficflow.pipeline import road_frame_from
from trafficflow.speed import MPS_TO_KMH, estimate_speed
from trafficflow.tracks import Track

FPS = 10.0


@pytest.fixture(scope="module")
def frame():
    where = Paths()
    return road_frame_from(load_site(where.config / "site.yaml"),
                           load_calibration(where.calibration))


def image_row_for(frame, y_m: float, column: float = 160.0) -> float:
    """The image row whose road position is ``y_m`` metres along the road."""
    rows = np.linspace(1.0, 239.0, 8000)
    points = np.column_stack([np.full_like(rows, column), rows])
    y = frame.to_road(points)[:, 1]
    order = np.argsort(y)
    return float(np.interp(y_m, y[order], rows[order]))


def image_col_for(frame, row: float, x_m: float) -> float:
    """The image column at ``row`` whose lateral road position is ``x_m``.

    Solved exactly rather than searched: for a fixed row the lateral map is linear in
    the column, so two samples pin it.
    """
    a, b = (frame.to_road([[c, row]])[0, 0] for c in (0.0, 100.0))
    return 100.0 * (x_m - a) / (b - a)


def track_at(frame, speed_kmh: float, start_m: float, n_frames: int = 30,
             lateral_m: float = 9.0) -> Track:
    """A vehicle holding ``speed_kmh`` from ``start_m``, built in image pixels."""
    metres_per_frame = speed_kmh / MPS_TO_KMH / FPS
    boxes = []
    for i in range(n_frames):
        row = image_row_for(frame, start_m + metres_per_frame * i)
        column = image_col_for(frame, row, lateral_m)
        half = 6.0 if row > 150 else 2.0        # distant vehicles are smaller
        boxes.append([column - half, row - half, column + half, row])
    return Track(
        track_id=1,
        frames=np.arange(n_frames),
        boxes=np.asarray(boxes, dtype=float),
        classes=np.full(n_frames, 2),
        confidences=np.full(n_frames, 0.9),
    )


# -- what the scale actually is --------------------------------------------------


def test_the_pixel_scale_spans_the_range_the_module_documents(frame):
    """Pins the numbers the speed module's reasoning is built on.

    Its docstring used to claim 12.6 px/m near the camera, which would make one pixel
    8 cm. The real figure at the near edge of the zone is about 3.8 px/m -- one pixel
    is 26 cm, three times worse -- and at the far edge one pixel covers 1.8 m.
    """
    near_row = image_row_for(frame, 60.0)
    far_row = image_row_for(frame, 156.0)

    near = float(frame.metres_per_pixel([[160.0, near_row]])[0])
    far = float(frame.metres_per_pixel([[160.0, far_row]])[0])

    assert near == pytest.approx(0.26, abs=0.03)      # ~3.8 px/m, not 12.6
    assert far == pytest.approx(1.78, abs=0.10)
    assert far / near > 5.0                            # why one fixed tolerance fails


def test_a_metre_tolerance_would_be_under_one_pixel_at_the_far_edge(frame):
    """The reason the tolerance is derived rather than written down.

    The old fixed 1.5 m sat below the size of a single pixel out here, so a distant
    track could not accumulate inliers however well it was tracked.
    """
    far_row = image_row_for(frame, 156.0)
    assert float(frame.metres_per_pixel([[160.0, far_row]])[0]) > 1.5


# -- measuring out there ----------------------------------------------------------


def test_a_vehicle_at_the_far_end_of_the_zone_is_measured_and_trusted(frame):
    """The case the fixed tolerance quietly discarded."""
    estimate = estimate_speed(track_at(frame, 40.0, start_m=130.0), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh == pytest.approx(40.0, rel=0.05)
    assert estimate.is_reliable


def test_a_vehicle_near_the_camera_is_measured_too(frame):
    estimate = estimate_speed(track_at(frame, 96.56, start_m=62.0), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh == pytest.approx(96.56, rel=0.05)
    assert estimate.is_reliable


def test_the_tolerance_grows_with_distance(frame):
    """Same road, same fitting, different pixel size -- so a different bound."""
    near = estimate_speed(track_at(frame, 60.0, start_m=62.0), frame, FPS)
    far = estimate_speed(track_at(frame, 60.0, start_m=130.0), frame, FPS)
    assert near is not None and far is not None
    assert far.tolerance_m > near.tolerance_m


def test_a_vehicle_stopped_at_the_far_end_still_reads_as_stopped(frame):
    """Jitter out here is metres of road, so the jitter bound has to scale with it."""
    rng = np.random.default_rng(0)
    row = image_row_for(frame, 140.0)
    column = image_col_for(frame, row, 9.0)
    rows = row + rng.normal(0.0, 0.7, 30)          # under a pixel of box wobble
    boxes = np.column_stack([
        np.full(30, column - 6.0), rows - 6.0, np.full(30, column + 6.0), rows,
    ])
    track = Track(1, np.arange(30), boxes, np.full(30, 2), np.full(30, 0.9))

    estimate = estimate_speed(track, frame, FPS)
    assert estimate is not None
    assert estimate.is_stationary
    assert estimate.is_reliable
    assert estimate.speed_kmh < 12.0
