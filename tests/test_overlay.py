"""Vehicles are drawn; the road surface is left as the camera saw it."""

from __future__ import annotations

import numpy as np

from trafficflow.overlay import annotate_frame


def test_a_vehicle_box_is_drawn_and_the_road_is_untouched():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    rows = np.array([[1, 7, 2, 0.9, 40, 40, 60, 60]], dtype=float)   # frame, id, car, conf, box
    canvas = annotate_frame(image, rows, speeds={}, scale=1)
    assert canvas[50, 40].any(), "the box edge"
    assert not canvas[90, 10].any(), "nothing painted on the empty road"
