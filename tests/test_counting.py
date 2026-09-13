"""Counting: a vehicle counts once it has been inside the measurement zone for 5 frames."""

from __future__ import annotations

import numpy as np

from synthetic_road import FPS
from trafficflow.counting import count_vehicles
from trafficflow.tracks import Track


def track(speed_kmh: float, n_frames: int, start_m: float = 10.0, lateral_m: float = 5.0,
          track_id: int = 1, cls: int = 2) -> Track:
    index = np.arange(n_frames)
    y = start_m + speed_kmh / 3.6 / FPS * index
    boxes = np.column_stack([np.full(n_frames, lateral_m - 1), y - 2, np.full(n_frames, lateral_m + 1), y])
    return Track(track_id, index, boxes, np.full(n_frames, cls), np.full(n_frames, 0.9))


def test_a_slow_vehicle_that_never_reaches_mid_zone_is_still_counted(frame):
    """The old line-crossing rule counted 2 vehicles in a whole heavy clip."""
    assert count_vehicles([track(11.0, 50)], frame).total == 1


def test_a_short_fragment_is_not_counted(frame):
    assert count_vehicles([track(90.0, 4)], frame).total == 0


def test_a_vehicle_outside_the_zone_is_not_counted(frame):
    assert count_vehicles([track(90.0, 30, start_m=400.0)], frame).total == 0


def test_class_and_lane_are_recorded(frame):
    counts = count_vehicles([track(60.0, 20, cls=7, lateral_m=5.0), track(60.0, 20, track_id=2)], frame)
    assert counts.by_class["truck"] == 1 and counts.by_class["car"] == 1
    assert counts.per_lane[2] == 2
    assert counts.tracks_seen == 2
