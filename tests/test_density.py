"""Density averages over every frame, so empty frames count as zero."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_road import LANE_COUNT, ZONE_M
from trafficflow.config import load_site
from trafficflow.density import METRES_PER_MILE, estimate_density
from trafficflow.tracks import TrackTable


def test_frames_without_vehicles_lower_the_density(frame):
    # one car, inside the zone, in frames 1..5 of a 10-frame clip
    rows = np.array([[n, 1, 2, 0.9, 4.0, 48.0, 6.0, 50.0] for n in range(1, 6)], dtype=float)
    los = load_site().level_of_service

    estimate = estimate_density(TrackTable(rows), frame, los, n_frames=10)

    assert estimate.vehicles_per_frame == pytest.approx(0.5)
    lane_miles = ZONE_M / METRES_PER_MILE * LANE_COUNT
    assert estimate.density_pc_per_mi_per_ln == pytest.approx(0.5 / lane_miles)


def test_without_n_frames_the_span_of_the_table_is_used(frame):
    rows = np.array([[n, 1, 2, 0.9, 4.0, 48.0, 6.0, 50.0] for n in (1, 10)], dtype=float)
    estimate = estimate_density(TrackTable(rows), frame, load_site().level_of_service)
    assert estimate.vehicles_per_frame == pytest.approx(0.2)
