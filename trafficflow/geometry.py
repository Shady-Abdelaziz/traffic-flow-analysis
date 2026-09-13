"""Mapping image pixels to positions on the road surface.

A single fixed camera cannot recover metres on its own -- that is a theorem, not a
limitation of the method. Here the painted road edges give the horizon and, with the
assumed road width, the camera height; UniDepth v2 gives the focal length. See
``01_calibration.ipynb``.

Given that, a **homography** relates the image plane to the road plane. The relation
holds only for points that genuinely lie on the road surface, which drives the single
most important rule in this module:

    Track a vehicle by the **bottom-centre of its box**, never the centroid.

The bottom edge is the tyre contact patch, which is on the road. The centroid floats
roughly a metre above it, and that metre is projected into a longitudinal error that
grows with distance -- tens of metres at the far end of the measurement zone.

The road-plane coordinate system used throughout:

    x  metres to the right of the traced yellow left-edge line
    y  metres away from the camera along the road

**The road bends, and this module accounts for it.** Calibration traces the yellow left
edge and measures how far it sweeps across the visible road -- most of a lane width.
Treating that edge as straight would put a vehicle in the wrong lane over much of the
zone. The figure is measured on each calibration run rather than quoted here: a number
written into a docstring is a number nobody rechecks.

A homography alone returns lateral distance from the *camera axis*. Because the road
curves, the left edge sits at a different lateral distance at every image row, so
:meth:`RoadFrame.to_road` subtracts the edge's own road position at each point's row.
That is what makes ``x`` mean "metres right of the left edge" rather than "metres right
of the axis". The edge comes from ``calibration.yaml``'s ``left_edge_poly``; a frame
built without one falls back to the straight-road assumption, which is what the unit
tests use.

Speeds read the longitudinal coordinate only and are unaffected either way. Lane
occupancy and lane-change detection depend on this correction entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "Homography",
    "RoadFrame",
    "ground_point",
    "ground_points",
]


def ground_point(box: Sequence[float]) -> tuple[float, float]:
    """Return the point of a bounding box that lies on the road.

    Parameters
    ----------
    box:
        ``(x1, y1, x2, y2)`` in pixels, y increasing downward.

    Returns
    -------
    tuple of float
        Bottom-centre of the box -- the tyre contact patch.

    Notes
    -----
    Using the centroid instead is the classic error in monocular speed estimation. A
    homography is only valid for coplanar points, and the centroid is about a metre
    above the road plane, so it back-projects to a position further from the camera
    than the vehicle actually is.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def ground_points(boxes: np.ndarray) -> np.ndarray:
    """Vectorised :func:`ground_point` over an ``(N, 4)`` array of boxes."""
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]])


@dataclass(frozen=True)
class Homography:
    """A 3x3 projective map between the image plane and the road plane.

    Attributes
    ----------
    matrix:
        Maps image pixels to road metres. Stored as ``float64``; the inverse is
        computed once on construction so repeated use costs nothing.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "_inverse", np.linalg.inv(matrix))

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_correspondences(
        cls,
        image_points: Sequence[Sequence[float]],
        road_points: Sequence[Sequence[float]],
    ) -> "Homography":
        """Fit the map from at least four matched points.

        Parameters
        ----------
        image_points:
            ``(N, 2)`` pixel coordinates, N >= 4.
        road_points:
            ``(N, 2)`` road-plane coordinates in metres, same order.

        Raises
        ------
        ValueError
            If fewer than four points are supplied, or the configuration is
            degenerate -- typically because three of the points are collinear, which
            happens if all four are taken from the same painted line.
        """
        image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        road = np.asarray(road_points, dtype=np.float64).reshape(-1, 2)
        if len(image) != len(road):
            raise ValueError(f"{len(image)} image points but {len(road)} road points")
        if len(image) < 4:
            raise ValueError(f"a homography needs at least 4 points, got {len(image)}")

        if len(image) == 4:
            matrix = cv2.getPerspectiveTransform(
                image.astype(np.float32), road.astype(np.float32)
            )
        else:
            # More than four: least squares, with RANSAC to survive a mis-clicked point.
            matrix, _ = cv2.findHomography(image, road, method=cv2.RANSAC, ransacReprojThreshold=0.25)

        if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError(
                "degenerate point configuration -- the four points must not be "
                "collinear; take two from each of two different painted lines"
            )
        return cls(matrix)

    @classmethod
    def from_lane_lines(
        cls,
        near_points: Sequence[Sequence[float]],
        far_points: Sequence[Sequence[float]],
        lateral_offsets_m: Sequence[float],
        near_distance_m: float,
        far_distance_m: float,
    ) -> "Homography":
        """Build the map from painted lines seen at two rows of the image.

        A general four-point construction, kept for tests and for any future
        calibration that has surveyed line positions to work from. The notebook does
        not use it: it builds the homography analytically from the fitted road plane,
        so no lateral offsets need to be known in advance.

        Parameters
        ----------
        near_points, far_points:
            Pixel positions where each painted line crosses the near and far
            reference rows, in the same order as ``lateral_offsets_m``.
        lateral_offsets_m:
            Distance of each line from the yellow left edge, in metres, e.g.
            ``(0.0, 3.0, 15.0)``.
        near_distance_m, far_distance_m:
            Longitudinal distance of the two reference rows from the camera.
        """
        if len(near_points) != len(lateral_offsets_m) or len(far_points) != len(lateral_offsets_m):
            raise ValueError(
                f"expected {len(lateral_offsets_m)} points per row, got "
                f"{len(near_points)} near and {len(far_points)} far"
            )
        image = [*near_points, *far_points]
        road = (
            [(offset, near_distance_m) for offset in lateral_offsets_m]
            + [(offset, far_distance_m) for offset in lateral_offsets_m]
        )
        return cls.from_correspondences(image, road)

    # -- use ------------------------------------------------------------------

    def to_road(self, points_px: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` pixel coordinates to road-plane metres."""
        return self._apply(points_px, self.matrix)

    def to_image(self, points_m: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` road-plane metres back to pixel coordinates."""
        return self._apply(points_m, self._inverse)  # type: ignore[attr-defined]

    @staticmethod
    def _apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


@dataclass(frozen=True)
class RoadFrame:
    """The road plane, its lane boundaries, and the zone worth measuring in.

    Attributes
    ----------
    homography:
        Image-to-road map.
    lane_width_m:
        Width of one lane -- the measured road width divided by the lane count,
        since the interior lane lines cannot be resolved to measure them apart.
    lane_count:
        Number of lanes, including the HOV lane.
    measurement_zone_m:
        Far end of the measurement zone, metres along the road.
    zone_start_m:
        Near end of the zone. The bottom of the image is already tens of metres from
        the camera, so the zone does not start at 0.
    left_edge_poly:
        Quadratic in the image row giving the column of the yellow left-edge line,
        highest power first, from ``calibration.yaml``. ``None`` assumes a straight
        road, which is right for synthetic test fixtures and wrong for this site.
    pixel_shift:
        ``(dx, dy)`` added to image points before mapping. The camera moved between
        the two recording days; this moves a day-1 pixel onto the calibrated day-2 image.
    """

    homography: Homography
    lane_width_m: float
    lane_count: int
    measurement_zone_m: float
    left_edge_poly: tuple[float, ...] | None = None
    zone_start_m: float = 0.0
    pixel_shift: tuple[float, float] = (0.0, 0.0)

    @property
    def travel_width_m(self) -> float:
        """Total width of the travelled way."""
        return self.lane_count * self.lane_width_m

    @property
    def zone_length_m(self) -> float:
        """Length of road inside the measurement zone."""
        return self.measurement_zone_m - self.zone_start_m

    def lane_index(self, x_m: np.ndarray | float) -> np.ndarray:
        """Lane number for a lateral position, 1 at the left edge.

        Returns 0 for anything left of the roadway and ``lane_count + 1`` for
        anything right of it, so off-road detections are visible rather than being
        silently clamped into a real lane.
        """
        x = np.atleast_1d(np.asarray(x_m, dtype=np.float64))
        index = np.floor(x / self.lane_width_m).astype(int) + 1
        index[x < 0] = 0
        index[x >= self.travel_width_m] = self.lane_count + 1
        return index

    def in_zone(self, road_points: np.ndarray) -> np.ndarray:
        """Boolean mask of points on the roadway and within the measurement zone."""
        points = np.asarray(road_points, dtype=np.float64).reshape(-1, 2)
        x, y = points[:, 0], points[:, 1]
        on_road = (x >= 0) & (x < self.travel_width_m)
        return on_road & (y >= self.zone_start_m) & (y <= self.measurement_zone_m)

    def left_edge_x(self, rows: np.ndarray) -> np.ndarray:
        """Image column of the yellow left-edge line at each image row."""
        if self.left_edge_poly is None:
            raise ValueError("this RoadFrame was built without a left_edge_poly")
        a, b, c = self.left_edge_poly
        rows = np.asarray(rows, dtype=np.float64)
        return a * rows * rows + b * rows + c

    def to_road(self, image_points: np.ndarray) -> np.ndarray:
        """Map image points to road coordinates measured from the left edge.

        The homography by itself gives lateral distance from the camera axis. Since
        the road curves, the left edge lies at a different lateral distance at every
        image row, so its position at each point's own row is subtracted here.

        The correction pushes the edge through the *same* homography rather than
        re-deriving the pinhole relation, so the two cannot drift apart.
        """
        points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2) + self.pixel_shift
        road = self.homography.to_road(points)
        if self.left_edge_poly is None:
            return road
        rows = points[:, 1]
        edge = np.column_stack([self.left_edge_x(rows), rows])
        road[:, 0] -= self.homography.to_road(edge)[:, 0]
        return road

    def metres_per_pixel(self, image_points: np.ndarray) -> np.ndarray:
        """Along-road metres covered by one image row, at each given image point.

        The perspective map is strongly non-uniform: at the near edge of the zone one
        row of pixels is about a quarter of a metre of road, and at the far edge it is
        well over a metre. A tolerance written as a single distance in metres therefore
        means something quite different at the two ends -- generous near the camera and
        smaller than one pixel far from it, which silently rejects distant tracks.

        Callers that need a tolerance derive it from here instead. Measured by pushing
        the point and the point one row below it through the same map, so it cannot
        drift away from whatever :meth:`to_road` actually does.
        """
        points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        below = points + np.array([0.0, 1.0])
        return np.abs(self.to_road(below)[:, 1] - self.to_road(points)[:, 1])

    def project_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map detection boxes straight to road-plane positions.

        Convenience for the common path: take each box's tyre contact patch and put
        it on the road plane in one call, so no caller has to remember which corner
        of the box is the valid one, and every caller gets the curve correction.
        """
        return self.to_road(ground_points(boxes))
