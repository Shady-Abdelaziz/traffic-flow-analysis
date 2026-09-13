"""Tests for speed estimation.

The road plane is mocked out with an identity homography, so one pixel is one metre and
a vehicle's road position is whatever the test puts there. That keeps these tests about
the *fitting*, which is what this module actually does -- the pixel-to-metre mapping has
its own tests in ``test_geometry.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_road import FPS, ZONE_M
from trafficflow.speed import (
    MAX_TOLERANCE_M,
    MIN_FRAMES,
    MIN_R_SQUARED,
    MIN_TOLERANCE_M,
    MPS_TO_KMH,
    SpeedEstimate,
    estimate_speed,
    estimate_speeds,
    summarise,
)
from trafficflow.tracks import Track


def straight_track(
    speed_kmh: float,
    n_frames: int = 30,
    start_m: float = 10.0,
    lateral_m: float = 5.0,
    track_id: int = 1,
    cls: int = 2,
) -> Track:
    """A vehicle holding ``speed_kmh`` exactly, in a straight line down one lane."""
    index = np.arange(n_frames)
    metres_per_frame = speed_kmh / MPS_TO_KMH / FPS
    y = start_m + metres_per_frame * index

    # Boxes are 2 m wide and 2 m tall; the ground point is the bottom-centre, so the
    # lateral position is the box centre and the longitudinal position is its bottom.
    left = np.full(n_frames, lateral_m - 1.0)
    right = np.full(n_frames, lateral_m + 1.0)
    boxes = np.column_stack([left, y - 2.0, right, y])
    return Track(
        track_id=track_id,
        frames=index,
        boxes=boxes,
        classes=np.full(n_frames, cls),
        confidences=np.full(n_frames, 0.9),
    )


def curving_track(
    speed_kmh: float, radius_m: float = 120.0, n_frames: int = 30, start_m: float = 10.0
) -> Track:
    """A vehicle holding ``speed_kmh`` around a right-hand bend of ``radius_m``.

    Positions are placed along a true circular arc at constant *arc* speed, which is
    what a real vehicle does. Measuring only the longitudinal coordinate would read
    slow here, because part of the travel goes sideways.
    """
    index = np.arange(n_frames)
    metres_per_frame = speed_kmh / MPS_TO_KMH / FPS
    travelled = start_m + metres_per_frame * index

    angle = travelled / radius_m                 # arc length over radius
    y = radius_m * np.sin(angle)                 # along the chord
    x = radius_m * (1 - np.cos(angle))           # sideways, into the bend

    boxes = np.column_stack([x - 1.0, y - 2.0, x + 1.0, y])
    return Track(
        track_id=1, frames=index, boxes=boxes,
        classes=np.full(n_frames, 2), confidences=np.full(n_frames, 0.9),
    )


# -- the fit ---------------------------------------------------------------------


@pytest.mark.parametrize("speed_kmh", [40.0, 60.0, 96.56, 120.0])
def test_a_constant_speed_comes_back_exactly(frame, speed_kmh):
    estimate = estimate_speed(straight_track(speed_kmh), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh == pytest.approx(speed_kmh, abs=0.01)
    assert estimate.r_squared == pytest.approx(1.0, abs=1e-6)


def test_a_clean_track_is_reliable(frame):
    estimate = estimate_speed(straight_track(96.56), frame, FPS)
    assert estimate.n_points == 30
    assert estimate.n_inliers == 30
    assert estimate.is_reliable


def test_the_lane_is_the_one_the_vehicle_drove_in(frame):
    # 5.0 m from the left edge is inside lane 2, which spans 3.0 .. 6.0 m.
    assert estimate_speed(straight_track(80.0, lateral_m=5.0), frame, FPS).lane == 2
    assert estimate_speed(straight_track(80.0, lateral_m=1.0), frame, FPS).lane == 1


def test_the_vehicle_class_survives_the_fit(frame):
    assert estimate_speed(straight_track(80.0, cls=7), frame, FPS).vehicle_class == "truck"


# -- refusing to answer ----------------------------------------------------------


def test_too_few_frames_returns_none_rather_than_a_number(frame):
    """Below MIN_FRAMES the jitter dominates, so no estimate is the honest answer."""
    assert estimate_speed(straight_track(96.56, n_frames=MIN_FRAMES - 1), frame, FPS) is None


def test_a_vehicle_beyond_the_measurement_zone_is_not_measured(frame):
    """Past the zone the road is no longer flat and vehicles are a few pixels across."""
    assert estimate_speed(straight_track(96.56, start_m=ZONE_M + 50), frame, FPS) is None


def test_only_the_points_inside_the_zone_are_counted(frame):
    """A track entering the zone mid-way is fitted on the part that is inside it."""
    track = straight_track(96.56, n_frames=60, start_m=ZONE_M - 40)
    estimate = estimate_speed(track, frame, FPS)
    assert estimate is not None
    assert estimate.n_points < 60
    assert estimate.speed_kmh == pytest.approx(96.56, abs=0.01)


# -- robustness ------------------------------------------------------------------


def test_a_swapped_identity_does_not_move_the_speed(frame):
    """RANSAC fits the majority, so a handful of points from another vehicle are
    reported as outliers rather than averaged into the answer."""
    track = straight_track(96.56, n_frames=40)
    boxes = track.boxes.copy()
    boxes[15:19, [1, 3]] += 45.0          # four frames jump to another trajectory
    swapped = Track(track.track_id, track.frames, boxes, track.classes, track.confidences)

    estimate = estimate_speed(swapped, frame, FPS)
    assert estimate.speed_kmh == pytest.approx(96.56, abs=2.0)
    assert estimate.n_inliers < estimate.n_points     # the damage is visible, not hidden


def test_fitting_beats_differencing_two_frames(frame):
    """The claim the module is built on, measured rather than asserted.

    One pixel of box jitter is about 8 cm on this road. Over the 0.1 s between two
    frames that is nearly 3 km/h; fitting a line through three seconds of positions
    should reduce it by close to an order of magnitude.
    """
    rng = np.random.default_rng(0)
    jitter_m, truth = 0.08, 96.56
    fit_errors, difference_errors = [], []

    for _ in range(200):
        track = straight_track(truth, n_frames=30)
        boxes = track.boxes.copy()
        boxes[:, [1, 3]] += rng.normal(0.0, jitter_m, size=(len(boxes), 1))
        noisy = Track(track.track_id, track.frames, boxes, track.classes, track.confidences)

        fit_errors.append(abs(estimate_speed(noisy, frame, FPS).speed_kmh - truth))
        two_frame = abs(boxes[-1, 3] - boxes[-2, 3]) * FPS * MPS_TO_KMH
        difference_errors.append(abs(two_frame - truth))

    assert np.mean(fit_errors) < np.mean(difference_errors) / 5


# -- summarising -----------------------------------------------------------------


def reliable(speed_kmh: float) -> SpeedEstimate:
    return SpeedEstimate(
        track_id=1, speed_kmh=speed_kmh, r_squared=0.999, n_points=30, n_inliers=30,
        distance_m=50.0, mean_distance_m=60.0, vehicle_class="car", lane=2,
    )


def test_summarising_nothing_gives_nan_not_zero():
    """Zero would read as 'the traffic was stationary'. It was not measured at all."""
    summary = summarise([])
    assert summary["n_reliable"] == 0
    assert np.isnan(summary["median_kmh"])


def test_unreliable_estimates_are_excluded_from_the_summary():
    poor = SpeedEstimate(2, 300.0, 0.2, 30, 10, 5.0, 60.0, "car", 2)
    assert not poor.is_reliable
    summary = summarise([reliable(90.0), reliable(100.0), poor])
    assert summary["n_vehicles"] == 3
    assert summary["n_reliable"] == 2
    assert summary["median_kmh"] == pytest.approx(95.0)


def test_the_85th_percentile_is_reported(frame):
    """Traffic engineers describe a stream by its 85th percentile, so it is computed."""
    summary = summarise([reliable(speed) for speed in range(80, 101)])
    assert summary["p85_kmh"] == pytest.approx(97.0)
    assert summary["p15_kmh"] == pytest.approx(83.0)


def test_estimate_speeds_drops_the_tracks_that_cannot_support_one(frame):
    tracks = [
        straight_track(90.0, track_id=1),
        straight_track(90.0, n_frames=3, track_id=2),           # too short
        straight_track(90.0, start_m=ZONE_M + 80, track_id=3),  # outside the zone
    ]
    assert [estimate.track_id for estimate in estimate_speeds(tracks, frame, FPS)] == [1]


# -- the road bends right --------------------------------------------------------


def test_a_vehicle_going_round_a_bend_reads_its_true_speed(frame) -> None:
    """Part of a turning vehicle's travel is sideways, and it still counts.

    Tested at a deliberately tight 120 m radius. The error scales with curvature --
    reading the longitudinal coordinate alone under-reads by 5.1% here but under 0.2%
    on a gentle motorway bend -- so a gentle radius would pass either way and protect
    nothing. This road's own curvature is measured by calibration, not assumed here.
    """
    estimate = estimate_speed(curving_track(100.0, radius_m=120.0), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh == pytest.approx(100.0, rel=0.01)



def test_mean_distance_is_the_distance_along_the_road(frame):
    estimate = estimate_speed(straight_track(96.56), frame, FPS)
    y = 10.0 + 96.56 / MPS_TO_KMH / FPS * np.arange(30)
    assert estimate.mean_distance_m == pytest.approx(y.mean(), abs=0.01)


def test_a_steady_speed_round_a_bend_does_not_look_like_acceleration(frame) -> None:
    """Constant speed must still fit a straight line in time.

    If the curve leaked into the distance measure the fit would bend, which is how a
    wrong distance measure shows itself even when the slope looks plausible.
    """
    estimate = estimate_speed(curving_track(100.0, radius_m=120.0), frame, FPS)
    assert estimate is not None
    assert estimate.r_squared > 0.999


# -- vehicles that are not moving -------------------------------------------------


def stationary_track(
    n_frames: int = 30, at_m: float = 60.0, jitter_m: float = 0.15, seed: int = 0
) -> Track:
    """A vehicle stopped at ``at_m``, its box breathing by ``jitter_m`` each frame.

    This is what a queued vehicle looks like to the detector: no travel, and a ground
    point that wanders by a fraction of a box.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(n_frames)
    y = at_m + rng.normal(0.0, jitter_m, n_frames)
    x = 5.0 + rng.normal(0.0, jitter_m, n_frames)
    boxes = np.column_stack([x - 1.0, y - 2.0, x + 1.0, y])
    return Track(
        track_id=1, frames=index, boxes=boxes,
        classes=np.full(n_frames, 2), confidences=np.full(n_frames, 0.9),
    )


def test_a_stopped_vehicle_reads_as_stopped_rather_than_as_a_failure(frame):
    """The case that used to be thrown away, and it is the one congestion is made of.

    A stopped vehicle's positions are pure jitter, so the fitted line explains nothing
    and R-squared sits near zero. Judging it by fit quality rejects it -- which means
    the only speeds surviving a jam were from whatever was still moving, and the clip
    read faster than the road actually was.
    """
    estimate = estimate_speed(stationary_track(), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh < 2.0
    assert estimate.r_squared < MIN_R_SQUARED      # the old gate would have refused it
    assert estimate.is_stationary
    assert estimate.is_reliable                    # and the new one does not


def test_a_clip_where_nothing_moves_reports_zero_not_nothing(frame):
    """All-stopped clips used to summarise to NaN and drop out of the diagram fit."""
    tracks = [
        Track(track_id=i, frames=t.frames, boxes=t.boxes,
              classes=t.classes, confidences=t.confidences)
        for i, t in enumerate((stationary_track(seed=s) for s in range(4)), start=1)
    ]
    summary = summarise(estimate_speeds(tracks, frame, FPS))
    assert summary["n_reliable"] == 4
    assert summary["median_kmh"] < 2.0
    assert not np.isnan(summary["median_kmh"])


def test_a_crawling_vehicle_is_kept_on_its_own_merits(frame):
    """8 km/h is slow but real travel, so it passes on fit quality, not on the exception."""
    estimate = estimate_speed(straight_track(8.0), frame, FPS)
    assert estimate is not None
    assert estimate.speed_kmh == pytest.approx(8.0, abs=0.01)
    assert not estimate.is_stationary
    assert estimate.is_reliable


def test_a_moving_vehicle_with_a_broken_fit_is_still_refused(frame):
    """The stationary exception must not become a way in for genuinely bad fits."""
    rng = np.random.default_rng(1)
    track = straight_track(90.0, n_frames=30)
    boxes = track.boxes.copy()
    boxes[:, [1, 3]] += rng.normal(0.0, 12.0, size=(len(boxes), 1))   # nonsense positions
    broken = Track(track.track_id, track.frames, boxes, track.classes, track.confidences)

    estimate = estimate_speed(broken, frame, FPS)
    assert estimate is not None
    assert not estimate.is_stationary
    assert not estimate.is_reliable


# -- the tolerance follows the pixel size -----------------------------------------


def test_the_tolerance_is_recorded_with_the_estimate(frame):
    """It is the jitter bound `is_stationary` compares against, so it is not implicit."""
    estimate = estimate_speed(straight_track(90.0), frame, FPS)
    assert MIN_TOLERANCE_M <= estimate.tolerance_m <= MAX_TOLERANCE_M
