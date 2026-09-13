"""The image-to-road map is correct, and it stays correct where the road bends.

The database ships no measured speeds, and the road geometry is no longer looked up in
an inventory -- it is measured by calibration -- so there is no external truth to test
against. These tests check the two things that can be checked without one.

*Internal consistency*, on a synthetic camera whose answers are known by construction:
a point maps to road and back to the same pixel, the anchors return the offsets they
were defined as, and a road that only *looks* like it tapers comes back a constant width.

*The curve correction*, on the real calibration: a point on the traced left edge must
read as zero metres across at every row, which is the definition of the road x
coordinate and is false the moment the correction is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from artefacts import needs_calibration
from trafficflow.config import load_calibration
from trafficflow.geometry import Homography, RoadFrame, ground_point, ground_points

# A synthetic but realistic camera: looking down the road, so the roadway narrows
# toward the top of the frame exactly as it does in the footage.
NEAR_ROW, FAR_ROW = 230.0, 150.0
NEAR_LEFT, NEAR_RIGHT = 120.0, 300.0     # roadway edges on the near row, px
FAR_LEFT, FAR_RIGHT = 175.0, 230.0       # and on the far row -- much narrower
NEAR_DISTANCE_M, FAR_DISTANCE_M = 10.0, 90.0

# Dimensions of the synthetic road. Arbitrary round numbers chosen for this fixture --
# they are not a claim about the real site, whose width is measured by calibration.
TRAVEL_WIDTH_M = 15.0
LANE_COUNT = 5
LANE_WIDTH_M = TRAVEL_WIDTH_M / LANE_COUNT
ZONE_M = 150.0


@pytest.fixture(scope="module")
def homography() -> Homography:
    """Built the way calibration builds it: two painted lines seen at two rows."""
    return Homography.from_lane_lines(
        near_points=[(NEAR_LEFT, NEAR_ROW), (NEAR_RIGHT, NEAR_ROW)],
        far_points=[(FAR_LEFT, FAR_ROW), (FAR_RIGHT, FAR_ROW)],
        lateral_offsets_m=[0.0, TRAVEL_WIDTH_M],
        near_distance_m=NEAR_DISTANCE_M,
        far_distance_m=FAR_DISTANCE_M,
    )


# -- the ground point ----------------------------------------------------------


def test_ground_point_is_the_bottom_centre() -> None:
    """The tyre contact patch, not the centroid -- the homography only holds on the road.

    The box below has its centroid at row 70 and its contact patch at row 90, so the
    exact pair asserted here is also what distinguishes the two.
    """
    assert ground_point((100.0, 50.0, 140.0, 90.0)) == (120.0, 90.0)


def test_ground_points_matches_the_scalar_version() -> None:
    boxes = np.array([[100.0, 50.0, 140.0, 90.0], [10.0, 20.0, 30.0, 44.0]])
    expected = np.array([ground_point(b) for b in boxes])
    np.testing.assert_allclose(ground_points(boxes), expected)


# -- the map itself ------------------------------------------------------------


def test_round_trip_returns_the_original_pixel(homography: Homography) -> None:
    """image -> road -> image must land back within half a pixel."""
    pixels = np.array([[150.0, 220.0], [260.0, 200.0], [200.0, 170.0], [180.0, 235.0]])
    back = homography.to_image(homography.to_road(pixels))
    np.testing.assert_allclose(back, pixels, atol=0.5)


def test_known_widths_come_back_out(homography: Homography) -> None:
    """The anchors themselves must map to the offsets they were defined as."""
    corners = np.array(
        [[NEAR_LEFT, NEAR_ROW], [NEAR_RIGHT, NEAR_ROW], [FAR_LEFT, FAR_ROW], [FAR_RIGHT, FAR_ROW]]
    )
    road = homography.to_road(corners)
    np.testing.assert_allclose(road[:, 0], [0.0, TRAVEL_WIDTH_M, 0.0, TRAVEL_WIDTH_M], atol=0.01)
    np.testing.assert_allclose(
        road[:, 1], [NEAR_DISTANCE_M, NEAR_DISTANCE_M, FAR_DISTANCE_M, FAR_DISTANCE_M], atol=0.01
    )


def test_roadway_width_is_constant_with_distance(homography: Homography) -> None:
    """The road does not taper in reality, only in the image.

    Interpolating the edges at an intermediate row and mapping them must still give
    the full travel width. This is the property that makes speed measurable at all.
    """
    for t in (0.25, 0.5, 0.75):
        row = NEAR_ROW + t * (FAR_ROW - NEAR_ROW)
        left = NEAR_LEFT + t * (FAR_LEFT - NEAR_LEFT)
        right = NEAR_RIGHT + t * (FAR_RIGHT - NEAR_RIGHT)
        road = homography.to_road(np.array([[left, row], [right, row]]))
        assert road[1, 0] - road[0, 0] == pytest.approx(TRAVEL_WIDTH_M, abs=0.01)


def test_further_up_the_image_is_further_down_the_road(homography: Homography) -> None:
    """A sanity check on sign: smaller y in pixels means larger distance in metres."""
    near, far = homography.to_road(np.array([[200.0, NEAR_ROW], [200.0, FAR_ROW]]))
    assert far[1] > near[1]


def test_fewer_than_four_points_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        Homography.from_correspondences([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_collinear_points_are_rejected() -> None:
    """Four points taken from one painted line cannot define the map."""
    collinear = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0)]
    with pytest.raises(ValueError):
        Homography.from_correspondences(collinear, [(0, 0), (1, 0), (2, 0), (3, 0)])


def test_mismatched_point_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="image points but"):
        Homography.from_correspondences([(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (1, 0)])


# -- lanes ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def perspective_frame(homography: Homography) -> RoadFrame:
    """The realistic camera above, as a road frame.

    Named apart from the suite-wide ``frame`` fixture on purpose: that one is the
    identity road where a pixel is a metre, and these tests are about what a real
    perspective does to that assumption.
    """
    return RoadFrame(
        homography=homography,
        lane_width_m=LANE_WIDTH_M,
        lane_count=LANE_COUNT,
        measurement_zone_m=ZONE_M,
    )


def test_lane_boundaries_land_in_the_expected_lane(perspective_frame: RoadFrame) -> None:
    width = perspective_frame.lane_width_m
    centres = [(i + 0.5) * width for i in range(perspective_frame.lane_count)]
    np.testing.assert_array_equal(
        perspective_frame.lane_index(centres), np.arange(1, perspective_frame.lane_count + 1)
    )


def test_off_road_positions_are_flagged_not_clamped(perspective_frame: RoadFrame) -> None:
    """Silently clamping an off-road detection into lane 1 would corrupt occupancy."""
    assert perspective_frame.lane_index(-1.0)[0] == 0
    assert (
        perspective_frame.lane_index(perspective_frame.travel_width_m + 1.0)[0]
        == perspective_frame.lane_count + 1
    )


def test_measurement_zone_excludes_the_far_field(perspective_frame: RoadFrame) -> None:
    """Beyond the flat near field, distances are not trustworthy."""
    points = np.array([[5.0, 50.0], [5.0, 500.0], [-2.0, 50.0], [5.0, -1.0]])
    np.testing.assert_array_equal(perspective_frame.in_zone(points), [True, False, False, False])


def test_zone_start_excludes_road_nearer_than_the_image_shows(homography: Homography) -> None:
    """Density divides by the zone length, so the near end must be subtracted."""
    frame = RoadFrame(homography, LANE_WIDTH_M, LANE_COUNT, ZONE_M, zone_start_m=10.0)
    assert frame.zone_length_m == ZONE_M - 10.0
    np.testing.assert_array_equal(frame.in_zone([[5.0, 5.0], [5.0, 20.0]]), [False, True])


def test_pixel_shift_moves_a_day1_pixel_onto_the_calibrated_image(homography: Homography) -> None:
    shifted = RoadFrame(homography, LANE_WIDTH_M, LANE_COUNT, ZONE_M, pixel_shift=(-6.0, 5.0))
    plain = RoadFrame(homography, LANE_WIDTH_M, LANE_COUNT, ZONE_M)
    np.testing.assert_allclose(shifted.to_road([[166.0, 195.0]]), plain.to_road([[160.0, 200.0]]))


# -- the curve correction ------------------------------------------------------


@needs_calibration
def test_a_point_on_the_left_edge_is_zero_metres_across() -> None:
    """The definition of the road x coordinate, checked at several rows.

    Without the correction this is only true at whichever row the edge happened to
    be sampled at; the road curves, so everywhere else it drifts.
    """
    calibration = load_calibration()
    frame = RoadFrame(
        homography=Homography(calibration.homography),
        lane_width_m=LANE_WIDTH_M,
        lane_count=LANE_COUNT,
        measurement_zone_m=ZONE_M,
        left_edge_poly=tuple(calibration.left_edge_poly),
    )
    rows = np.array([110.0, 130.0, 150.0, 163.0])
    on_edge = np.column_stack([frame.left_edge_x(rows), rows])
    np.testing.assert_allclose(frame.to_road(on_edge)[:, 0], 0.0, atol=1e-9)


@needs_calibration
def test_ignoring_the_curve_costs_most_of_a_lane() -> None:
    """The reason the correction exists, stated as a number rather than a claim.

    Measured across the measurement zone, which is the span that matters -- not the
    narrower range of rows the left edge was fitted over.
    """
    calibration = load_calibration()
    homography = Homography(calibration.homography)

    # Rows bounding the measurement zone, taken by pushing the two ends of the visible
    # road back through the homography. Re-deriving them from f*h/(row - y0) would put a
    # second, approximate geometry beside the exact one the calibration already carries,
    # and would need the near distance hard-coded here to stay in step with it.
    near_m, far_m = min(calibration.visible_road_m), max(calibration.visible_road_m)
    rows = homography.to_image(np.array([[0.0, far_m], [0.0, near_m]]))[:, 1]
    uncorrected = homography.to_road(
        np.column_stack([[calibration.left_edge_x(r) for r in rows], rows])
    )[:, 0]
    sweep = abs(uncorrected[1] - uncorrected[0])

    assert sweep > 0.7 * LANE_WIDTH_M, (
        f"the left edge sweeps only {sweep:.2f} m across the zone; if this ever falls "
        f"near zero the calibration changed and the correction may be unnecessary"
    )
