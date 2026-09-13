"""Vehicle speed from tracked positions on the road plane.

The measurement is: take a vehicle's ground point in each frame, map it to metres on the
road, and fit a straight line to distance against time. The slope is the speed.

**Why a fit over many frames rather than a difference between two.** At 10 fps the gap
between consecutive frames is 0.1 s. At the near edge of the measurement zone the scale
is about 3.8 px/m, so one pixel of detector jitter is roughly 26 cm; over 0.1 s that is
2.6 m/s, nearly 10 km/h of error from a single pixel. At the far edge of the zone it is
far worse: one pixel there covers about 1.8 m of road, because the same perspective that
makes distant vehicles small also stretches every pixel across more ground. Fitting a
line through thirty frames spreads that jitter over three seconds instead, and the random
part of it averages out. The difference between the two approaches is roughly an order of
magnitude in accuracy.

That 3.8-to-1.8 spread is also why the tolerances here are derived from
:meth:`~trafficflow.geometry.RoadFrame.metres_per_pixel` rather than written as fixed
distances -- a single number in metres is either generous near the camera or narrower
than one pixel at the far edge, and the second of those quietly discards exactly the
distant tracks that matter for a congested scene.

**Why RANSAC rather than least squares.** A tracker occasionally swaps identity between
two nearby vehicles, which puts a handful of points from a different trajectory into the
series. Least squares would quietly split the difference and report a speed belonging to
neither vehicle. RANSAC fits the majority and reports how many points disagreed, so the
failure is visible instead of averaged in.

Speeds are reported in km/h, the unit of the posted limit this road is judged against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficflow.geometry import RoadFrame, ground_points
from trafficflow.tracks import Track

__all__ = [
    "SpeedEstimate",
    "MPS_TO_KMH",
    "MIN_FRAMES",
    "MIN_R_SQUARED",
    "estimate_speed",
    "estimate_speeds",
    "summarise",
]


#: metres per second to kilometres per hour
MPS_TO_KMH = 3.6

#: A track must be seen in at least this many frames before a speed is fitted. Ten
#: frames is one second, long enough that detector jitter is no longer the dominant term.
MIN_FRAMES = 10

#: Fit quality a *moving* vehicle must reach before its speed is pooled. A vehicle
#: holding roughly constant speed gives above 0.99, so 0.90 leaves room for genuine
#: acceleration while still rejecting a broken track.
MIN_R_SQUARED = 0.90

#: Detector jitter, in pixels. The bottom edge of a YOLO box wanders a pixel or two
#: between frames even on a parked vehicle, so a residual inside this is noise rather
#: than evidence that the point belongs to a different track.
JITTER_PX = 2.0

#: Floor and ceiling on the residual tolerance, in metres. The floor keeps it from
#: collapsing near the camera, where a pixel is a quarter of a metre. The ceiling keeps
#: it below one frame of travel -- a vehicle at the limit covers 2.7 m per frame -- so a
#: swapped identity still shows up as outliers instead of being absorbed into the fit.
MIN_TOLERANCE_M = 0.5
MAX_TOLERANCE_M = 2.0


@dataclass(frozen=True, slots=True)
class SpeedEstimate:
    """One vehicle's speed, with the evidence behind it.

    Attributes
    ----------
    speed_kmh:
        Fitted speed, always non-negative. Direction of travel is deliberately *not*
        encoded in the sign: the distance axis comes from the principal direction of
        the track's own positions, and the sign of a principal direction is arbitrary,
        so a negative slope here would carry no information about which way the vehicle
        was going. A track that swapped identity shows up in ``n_inliers`` instead,
        which is a real signal rather than an artefact of the decomposition.
    r_squared:
        Fit quality. A vehicle at roughly constant speed gives above 0.99; a low value
        means genuine acceleration, a broken track, or a vehicle that barely moved --
        see :attr:`is_reliable`.
    n_points, n_inliers:
        Points used, and how many the robust fit accepted. A large gap is the signal
        that identity was swapped mid-track.
    distance_m:
        Ground covered between the first and last accepted point.
    mean_distance_m:
        Average distance from the camera over the track, so speeds can be grouped by
        how far away they were measured.
    tolerance_m:
        The residual tolerance this track was fitted with, derived from the pixel
        size where it was seen. Doubles as the jitter bound below which a displacement
        is indistinguishable from detector noise.
    """

    track_id: int
    speed_kmh: float
    r_squared: float
    n_points: int
    n_inliers: int
    distance_m: float
    mean_distance_m: float
    vehicle_class: str
    lane: int
    tolerance_m: float = 0.0

    @property
    def is_stationary(self) -> bool:
        """Whether the vehicle demonstrably did not move.

        Total displacement within the jitter bound means every millimetre of apparent
        travel could have come from the detector box breathing. The honest reading of
        that is "this vehicle was stopped", not "this measurement failed".
        """
        return self.distance_m <= self.tolerance_m

    @property
    def is_reliable(self) -> bool:
        """Whether this estimate should be pooled into clip-level statistics.

        A moving vehicle earns trust three ways: enough points, a fit that actually
        describes them, and most of those points agreeing with the fit.

        A *stopped* vehicle is the exception, and it matters. Its positions are pure
        jitter, so the fitted line explains nothing and R-squared sits near zero -- the
        moving-vehicle test rejects it. That rejection is backwards: it throws away
        precisely the vehicles that define congestion, so jammed clips reported speeds
        drawn only from whatever was still moving, and a clip where nothing moved at
        all reported no speed and dropped out of the fundamental diagram entirely.
        When the displacement is inside the jitter bound the near-zero speed is the
        measurement, so the fit-quality test does not apply.
        """
        if self.n_points < MIN_FRAMES or self.n_inliers < 0.7 * self.n_points:
            return False
        return self.is_stationary or self.r_squared >= MIN_R_SQUARED


def _fit_robust(
    times: np.ndarray, distances: np.ndarray, tolerance_m: float
) -> tuple[float, float, np.ndarray]:
    """Fit ``distance = speed * time + offset``, resisting a minority of bad points.

    Parameters
    ----------
    tolerance_m:
        Residual beyond which a point is treated as not belonging to the track. Comes
        from :func:`_tolerance_for`, so it reflects the pixel size where this
        particular vehicle was seen rather than a single figure for the whole image.

    Returns
    -------
    tuple
        ``(slope_mps, r_squared, inlier_mask)``.

    Notes
    -----
    Uses scikit-learn's RANSAC when available and falls back to least squares
    otherwise, so the module still works in a minimal environment -- the fallback is
    less robust, not wrong.
    """
    try:
        from sklearn.linear_model import RANSACRegressor
    except ImportError:  # pragma: no cover - exercised only without scikit-learn
        slope, offset = np.polyfit(times, distances, 1)
        predicted = slope * times + offset
        inliers = np.abs(distances - predicted) <= tolerance_m
        return float(slope), _r_squared(distances, predicted), inliers

    estimator = RANSACRegressor(
        residual_threshold=tolerance_m,
        max_trials=100,
        random_state=0,
    )
    estimator.fit(times.reshape(-1, 1), distances)
    inliers = estimator.inlier_mask_
    slope = float(estimator.estimator_.coef_[0])
    predicted = estimator.predict(times.reshape(-1, 1))
    return slope, _r_squared(distances[inliers], predicted[inliers]), inliers


def _tolerance_for(frame: RoadFrame, image_points: np.ndarray) -> float:
    """Residual tolerance, in metres, for a track seen at these image points.

    :data:`JITTER_PX` pixels of box wobble, converted to metres at the median row the
    track was actually seen on, then held between :data:`MIN_TOLERANCE_M` and
    :data:`MAX_TOLERANCE_M`. The median rather than the mean so that a few frames at
    the very top of the zone, where a pixel is worth metres, cannot stretch the
    tolerance for the whole track.
    """
    scale = float(np.median(frame.metres_per_pixel(image_points)))
    return float(np.clip(JITTER_PX * scale, MIN_TOLERANCE_M, MAX_TOLERANCE_M))


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination, 0 when the data has no variance to explain."""
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total <= 0:
        return 0.0
    residual = float(np.sum((actual - predicted) ** 2))
    return max(0.0, 1.0 - residual / total)


def estimate_speed(track: Track, frame: RoadFrame, fps: float) -> SpeedEstimate | None:
    """Fit a speed to one track.

    Parameters
    ----------
    track:
        A vehicle's path in image coordinates.
    frame:
        Supplies the image-to-road map, the lane geometry and the measurement zone.
    fps:
        Frames per second, read from the video rather than assumed.

    Returns
    -------
    SpeedEstimate or None
        ``None`` when too few points fall inside the measurement zone -- beyond it the
        road is no longer flat and a vehicle is a handful of pixels, so a speed fitted
        there would be arithmetic without meaning.
    """
    pixels = ground_points(track.boxes)
    road = frame.to_road(pixels)
    inside = frame.in_zone(road)
    if inside.sum() < MIN_FRAMES:
        return None

    road = road[inside]
    times = track.frames[inside] / fps
    tolerance = _tolerance_for(frame, pixels[inside])

    # Distance travelled, measured along the vehicle's own direction of travel.
    #
    # The road bends right, so the longitudinal coordinate alone is not the distance --
    # a turning vehicle also moves sideways, and that movement is real travel. Taking
    # the principal direction of its own positions absorbs the curve's heading without
    # assuming anything about the radius.
    #
    # Cumulative arc length would be the obvious alternative and is a worse one: every
    # pixel of tracking jitter adds to a path length, never subtracts, so it biases
    # speed upward. Projecting onto one direction averages jitter out instead, exactly
    # as the line fit below does in time.
    centred = road - road.mean(axis=0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    distances = centred @ direction

    slope, r_squared, inliers = _fit_robust(times, distances, tolerance)
    lanes = frame.lane_index(road[:, 0])
    values, counts = np.unique(lanes, return_counts=True)

    return SpeedEstimate(
        track_id=track.track_id,
        speed_kmh=abs(slope) * MPS_TO_KMH,
        r_squared=r_squared,
        n_points=int(inside.sum()),
        n_inliers=int(inliers.sum()),
        distance_m=float(abs(distances[inliers][-1] - distances[inliers][0])) if inliers.any() else 0.0,
        mean_distance_m=float(road[:, 1].mean()),
        vehicle_class=track.vehicle_class,
        lane=int(values[counts.argmax()]),
        tolerance_m=tolerance,
    )


def estimate_speeds(
    tracks: list[Track], frame: RoadFrame, fps: float
) -> list[SpeedEstimate]:
    """Fit speeds to every track that supports one."""
    estimates = (estimate_speed(track, frame, fps) for track in tracks)
    return [estimate for estimate in estimates if estimate is not None]


def summarise(estimates: list[SpeedEstimate]) -> dict[str, float]:
    """Clip-level speed statistics, computed only from reliable estimates.

    The median is reported alongside the mean because a single swapped identity can
    produce an implausible speed that drags a mean but barely moves a median.

    Stopped vehicles count. They reach this pool through :attr:`SpeedEstimate.is_reliable`
    even though their fits explain nothing, because a stationary vehicle's near-zero
    speed is a measurement and not a failure. Excluding them biased every congested
    clip upward, since what remained in the pool was whatever was still moving.

    Returns
    -------
    dict
        ``n_vehicles``, ``n_reliable``, ``mean_kmh``, ``median_kmh``, ``p15_kmh``,
        ``p85_kmh``, ``std_kmh``. The 85th percentile is the figure traffic engineers
        conventionally use to describe the prevailing speed of a stream.
    """
    reliable = [estimate for estimate in estimates if estimate.is_reliable]
    if not reliable:
        return {
            "n_vehicles": len(estimates),
            "n_reliable": 0,
            "mean_kmh": float("nan"),
            "median_kmh": float("nan"),
            "p15_kmh": float("nan"),
            "p85_kmh": float("nan"),
            "std_kmh": float("nan"),
        }

    speeds = np.array([estimate.speed_kmh for estimate in reliable])
    return {
        "n_vehicles": len(estimates),
        "n_reliable": len(reliable),
        "mean_kmh": float(speeds.mean()),
        "median_kmh": float(np.median(speeds)),
        "p15_kmh": float(np.percentile(speeds, 15)),
        "p85_kmh": float(np.percentile(speeds, 85)),
        "std_kmh": float(speeds.std(ddof=1)) if len(speeds) > 1 else 0.0,
    }
