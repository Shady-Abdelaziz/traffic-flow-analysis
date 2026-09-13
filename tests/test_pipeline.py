"""Batch and live share one measurement function."""

from __future__ import annotations

import numpy as np
import pytest

from artefacts import needs_calibration
from synthetic_road import FPS, identity_road_frame
from trafficflow.config import load_calibration, load_site
from trafficflow.pipeline import measure, road_frame_from
from trafficflow.tracks import TrackTable


def rows_for(track_id: int, speed_kmh: float, lateral_m: float, n_frames: int = 30) -> list[list[float]]:
    rows = []
    for n in range(1, n_frames + 1):
        y = 10.0 + speed_kmh / 3.6 / FPS * (n - 1)
        rows.append([n, track_id, 2, 0.9, lateral_m - 1, y - 2, lateral_m + 1, y])
    return rows


def test_measure_gives_every_parameter_from_one_table():
    frame = identity_road_frame()
    table = TrackTable(np.array(rows_for(1, 90.0, 4.5) + rows_for(2, 100.0, 10.5), dtype=float))

    result = measure(table, frame, FPS, load_site().level_of_service, n_frames=30)

    assert result.counts.total == 2
    assert result.speed_summary["median_kmh"] == pytest.approx(95.0, abs=0.1)
    assert result.density.vehicles_per_frame == pytest.approx(2.0)
    assert result.occupancy.occupancy[2] > 0
    assert result.lane_changes == []


@needs_calibration
def test_the_measurement_zone_reaches_the_bottom_of_the_frame():
    """A vehicle touching the last row of the picture is measured, not thrown away.

    The near end of the zone used to come from ``visible_road_m``, which records where
    the calibration's tracing started -- about a metre inside the picture. Everything in
    the last rows of the frame, where vehicles are largest and clearest, fell outside the
    zone and was drawn but never counted. Both recording days are checked: day 1's frames
    sit a few pixels lower, so its zone has to start nearer still.
    """
    site, calibration = load_site(), load_calibration()

    for day in (1, 2):
        frame = road_frame_from(site, calibration, day)
        bottom_row = float(site.image.height - 1)
        patch = np.array([[site.image.width - 1.0, bottom_row]])   # on the roadway, last row

        assert frame.in_zone(frame.to_road(patch))[0], (
            f"day {day}: a contact patch on row {bottom_row:.0f} is outside the zone, "
            f"which starts at {frame.zone_start_m:.2f} m"
        )
