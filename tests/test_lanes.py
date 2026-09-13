"""Tests for lane occupancy and lane-change detection.

The point of interest is the debounce-and-hysteresis pair. A vehicle tracking straight
along a painted line will appear to cross it many times as its detection box breathes,
and a naive counter reports dozens of lane changes for a vehicle that never changed
lane. These tests hold that behaviour in place: real changes are found, jitter is not.

As in ``test_speed.py`` the homography is the identity, so metres are pixels and the
tests are about the lane logic rather than the projection.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_road import LANE_COUNT, LANE_WIDTH_M, lane_centre
from trafficflow.lanes import (
    DEBOUNCE_FRAMES,
    detect_lane_changes,
    lane_occupancy,
    summarise_changes,
)
from trafficflow.tracks import COLUMNS, Track, TrackTable


def track_at(x: np.ndarray, n_frames: int | None = None, track_id: int = 1) -> Track:
    """A vehicle whose lateral position follows ``x``, driving steadily away."""
    x = np.asarray(x, dtype=float)
    n = n_frames or len(x)
    index = np.arange(n)
    y = 10.0 + 2.7 * index                       # ~97 km/h at 10 fps
    boxes = np.column_stack([x - 1.0, y - 2.0, x + 1.0, y])
    return Track(
        track_id=track_id,
        frames=index,
        boxes=boxes,
        classes=np.full(n, 2),
        confidences=np.full(n, 0.9),
    )


def table_from(tracks: list[Track]) -> TrackTable:
    """Assemble a TrackTable from tracks, in the column order the CSV uses."""
    rows = []
    for track in tracks:
        for i in range(len(track.frames)):
            rows.append([
                track.frames[i], track.track_id, track.classes[i], track.confidences[i],
                *track.boxes[i],
            ])
    return TrackTable(np.asarray(rows, dtype=np.float64).reshape(-1, len(COLUMNS)))


# -- lane changes ----------------------------------------------------------------


def test_a_vehicle_holding_one_lane_reports_no_change(frame):
    x = np.full(40, lane_centre(3))
    assert detect_lane_changes(track_at(x), frame) == []


def test_jitter_across_a_boundary_is_not_a_lane_change(frame):
    """The false-positive this module exists to prevent.

    The vehicle sits exactly on the line between lanes 2 and 3 and wobbles a few
    centimetres either side of it for four seconds. Counting raw crossings would
    report roughly twenty lane changes. The correct answer is none.
    """
    rng = np.random.default_rng(0)
    boundary = 2 * LANE_WIDTH_M
    x = boundary + rng.normal(0.0, 0.12, size=40)

    raw_crossings = int(np.sum(np.diff(np.floor(x / LANE_WIDTH_M)) != 0))
    assert raw_crossings > 5, "the fixture must actually wobble across the line"
    assert detect_lane_changes(track_at(x), frame) == []


def test_a_real_lane_change_is_found_with_its_direction(frame):
    x = np.concatenate([np.full(15, lane_centre(3)), np.full(25, lane_centre(2))])
    changes = detect_lane_changes(track_at(x), frame)

    assert len(changes) == 1
    assert (changes[0].from_lane, changes[0].to_lane) == (3, 2)
    assert changes[0].direction == "left"
    assert changes[0].frame == pytest.approx(15, abs=2)


def test_a_change_to_the_right_is_labelled_right(frame):
    x = np.concatenate([np.full(15, lane_centre(2)), np.full(25, lane_centre(4))])
    changes = detect_lane_changes(track_at(x), frame)
    assert [change.direction for change in changes] == ["right"]


def test_two_changes_in_one_track_are_both_found(frame):
    x = np.concatenate([
        np.full(12, lane_centre(4)), np.full(12, lane_centre(3)), np.full(12, lane_centre(2)),
    ])
    changes = detect_lane_changes(track_at(x), frame)
    assert [(c.from_lane, c.to_lane) for c in changes] == [(4, 3), (3, 2)]


def test_a_change_not_held_long_enough_is_not_confirmed(frame):
    """A brief excursion into the next lane is a wobble, not a lane change."""
    x = np.concatenate([
        np.full(15, lane_centre(3)),
        np.full(DEBOUNCE_FRAMES - 2, lane_centre(2)),
        np.full(15, lane_centre(3)),
    ])
    assert detect_lane_changes(track_at(x), frame) == []


def test_a_track_shorter_than_the_debounce_window_is_skipped(frame):
    x = np.full(DEBOUNCE_FRAMES, lane_centre(3))
    assert detect_lane_changes(track_at(x), frame) == []


# -- occupancy -------------------------------------------------------------------


def test_a_lane_used_throughout_reads_as_fully_occupied(frame):
    table = table_from([track_at(np.full(30, lane_centre(2)))])
    occupancy = lane_occupancy(table, frame)

    assert occupancy.occupancy[2] == pytest.approx(1.0)
    assert occupancy.occupancy[1] == 0.0
    assert occupancy.vehicle_counts[2] == 1
    assert occupancy.busiest_lane == 2


def test_occupancy_of_an_empty_clip_is_zero_everywhere(frame):
    occupancy = lane_occupancy(TrackTable(np.empty((0, len(COLUMNS)))), frame)
    assert set(occupancy.occupancy) == set(range(1, LANE_COUNT + 1))
    assert occupancy.imbalance == 0.0


def test_distinct_vehicles_in_one_lane_are_counted_separately(frame):
    table = table_from([
        track_at(np.full(30, lane_centre(4)), track_id=1),
        track_at(np.full(30, lane_centre(4)), track_id=2),
    ])
    assert lane_occupancy(table, frame).vehicle_counts[4] == 2


def test_imbalance_is_zero_when_every_lane_is_equally_busy(frame):
    table = table_from([
        track_at(np.full(30, lane_centre(lane)), track_id=lane)
        for lane in range(1, LANE_COUNT + 1)
    ])
    occupancy = lane_occupancy(table, frame)
    assert occupancy.imbalance == pytest.approx(0.0)


def test_imbalance_rises_when_traffic_crowds_one_lane(frame):
    table = table_from([track_at(np.full(30, lane_centre(1)))])
    assert lane_occupancy(table, frame).imbalance == pytest.approx(1.0)


# -- summarising -----------------------------------------------------------------


def test_changes_are_reported_per_vehicle_not_only_as_a_total(frame):
    """A busier clip shows more changes simply for having more vehicles; dividing by
    the vehicle count is what makes two clips comparable."""
    x = np.concatenate([np.full(15, lane_centre(3)), np.full(25, lane_centre(2))])
    changes = detect_lane_changes(track_at(x), frame)

    summary = summarise_changes(changes, n_vehicles=4)
    assert summary["n_lane_changes"] == 1
    assert summary["changes_per_vehicle"] == pytest.approx(0.25)
    assert summary["changes_left"] == 1
    assert summary["changes_right"] == 0


def test_summarising_with_no_vehicles_does_not_divide_by_zero():
    assert summarise_changes([], n_vehicles=0)["changes_per_vehicle"] == 0.0


# -- the denominator: frames with nothing in them ---------------------------------


def sparse_table(present_frames: int, lane: int = 2) -> TrackTable:
    """One vehicle visible for ``present_frames`` frames and absent afterwards."""
    x = np.full(present_frames, lane_centre(lane))
    return table_from([track_at(x)])


def test_occupancy_counts_the_whole_clip_not_only_the_busy_frames(frame):
    """Light traffic is mostly empty frames, and they belong in the denominator.

    Inferring the denominator from the frames that happen to hold a detection makes
    every lane's occupancy the fraction of *sightings*, which is 1.0 for any lane the
    single visible vehicle used. A road with one car on it then reports the same
    occupancy as a road that is full -- and this number is the video stand-in for the
    inductive loop at this milepost, feeds ``lane_imbalance`` into the classifier, and
    draws the live lane bars.
    """
    table = sparse_table(present_frames=10)

    told = lane_occupancy(table, frame, n_frames=100)
    assert told.occupancy[2] == pytest.approx(0.10)
    assert told.n_frames == 100

    # Without being told, all it can see is the ten frames that had a detection.
    inferred = lane_occupancy(table, frame)
    assert inferred.occupancy[2] > told.occupancy[2]


def test_an_empty_clip_still_reports_the_length_it_was_given(frame):
    empty = lane_occupancy(TrackTable(np.empty((0, len(COLUMNS)))), frame, n_frames=53)
    assert empty.n_frames == 53
    assert all(value == 0.0 for value in empty.occupancy.values())


def test_a_clip_whose_detections_all_fall_outside_the_zone_is_empty_not_full(frame):
    """Outside the zone there is no occupancy, and the denominator is still the clip."""
    far = table_from([track_at(np.full(30, lane_centre(2)), track_id=1)])
    far.rows[:, [5, 7]] += 10_000.0                  # push every box past the zone
    occupancy = lane_occupancy(far, frame, n_frames=30)
    assert occupancy.n_frames == 30
    assert all(value == 0.0 for value in occupancy.occupancy.values())


def test_occupancy_never_exceeds_one_when_the_clip_length_is_known(frame):
    table = table_from([
        track_at(np.full(30, lane_centre(lane)), track_id=lane) for lane in (1, 2, 3)
    ])
    occupancy = lane_occupancy(table, frame, n_frames=30)
    assert all(0.0 <= value <= 1.0 for value in occupancy.occupancy.values())
