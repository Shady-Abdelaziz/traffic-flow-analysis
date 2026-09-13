"""Traffic density and Level of Service.

Density is how many vehicles occupy a length of road, per lane -- the quantity that
decides whether a road is working. Unlike a raw vehicle count it does not depend on how
long the clip is or how wide the road is, so a 5-second clip and an hour of footage are
directly comparable.

    k = vehicles / (zone length in miles x lanes)

Miles per lane rather than metres because the Highway Capacity Manual's Level of Service
thresholds -- the A-to-F grade this produces -- are defined in passenger cars per mile per
lane, and converting the thresholds instead of the measurement would make every published
comparison harder to check.

**Trucks are not counted as one vehicle.** A truck occupies far more road and accelerates
far more slowly than a car, so the HCM converts to *passenger car equivalents* before
grading. Skipping that step understates congestion on a road with freight traffic. How
much freight *this* road carries is not assumed here: the class mix is counted per clip by
:mod:`trafficflow.counting`, so the size of the correction is visible in the output rather
than taken on trust.

One honesty note that belongs in the report: a clip is 5.3 seconds, while the HCM's
thresholds were calibrated on 15-minute averages. These are therefore instantaneous
density snapshots, and they scatter more than a 15-minute figure would. The grade is
still meaningful; its precision is not what the letter alone implies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficflow.config import LevelOfService
from trafficflow.geometry import RoadFrame
from trafficflow.tracks import TrackTable

__all__ = ["PASSENGER_CAR_EQUIVALENT", "DensityEstimate", "estimate_density"]


#: Passenger-car equivalents per vehicle type, for a level freeway segment (HCM 6th
#: edition, Exhibit 12-26). Level-terrain factors are an assumption: the calibration's
#: ``grade_deg`` is the camera's pitch, not the road's grade, so it cannot confirm this.
PASSENGER_CAR_EQUIVALENT: dict[str, float] = {
    "car": 1.0,
    "bicycle": 1.0,      # HCM gives no freeway factor (bicycles are banned); counted as one car
    "motorcycle": 1.0,
    "bus": 2.0,
    "truck": 2.0,
    "unknown": 1.0,
}

#: metres in a mile
METRES_PER_MILE = 1609.344


@dataclass(frozen=True, slots=True)
class DensityEstimate:
    """Density of one clip and the Level of Service it implies.

    Attributes
    ----------
    vehicles_per_frame:
        Mean raw vehicle count inside the measurement zone.
    density_veh_per_mi_per_ln:
        Density counting every vehicle as one.
    density_pc_per_mi_per_ln:
        Density in passenger-car equivalents -- the figure the Level of Service is
        graded from.
    level_of_service:
        A (free flow) to F (breakdown).
    heavy_vehicle_fraction:
        Share of detections that were buses or trucks. Reported because it is what
        separates the two densities, and because freight share is itself an insight.
    """

    vehicles_per_frame: float
    density_veh_per_mi_per_ln: float
    density_pc_per_mi_per_ln: float
    level_of_service: str
    heavy_vehicle_fraction: float
    n_frames: int


def estimate_density(
    table: TrackTable,
    frame: RoadFrame,
    level_of_service: LevelOfService,
    n_frames: int | None = None,
) -> DensityEstimate:
    """Measure density over one clip.

    Only detections inside the measurement zone count. Beyond it recall falls off and a
    vehicle is a few pixels across, so including that range would mean counting fewer
    vehicles over a longer road and understating density exactly where it matters.

    Parameters
    ----------
    table:
        The clip's track table.
    frame:
        Road geometry -- supplies the zone length and lane count.
    level_of_service:
        HCM thresholds, from ``config/site.yaml``.
    n_frames:
        Frames the clip covers. Frames with no vehicle in the zone count as zero.
        Defaults to the span of the table, first to last frame.
    """
    if len(table) == 0:
        return DensityEstimate(0.0, 0.0, 0.0, "A", 0.0, n_frames or 0)

    covered = n_frames or int(table.frames.max() - table.frames.min() + 1)
    road = frame.project_boxes(table.boxes)
    inside = frame.in_zone(road)
    frames = table.frames[inside]
    if not inside.any():
        return DensityEstimate(0.0, 0.0, 0.0, "A", 0.0, covered)

    from trafficflow.tracks import VEHICLE_CLASSES

    classes = table.classes[inside]
    equivalents = np.array(
        [PASSENGER_CAR_EQUIVALENT[VEHICLE_CLASSES.get(int(c), "unknown")] for c in classes]
    )
    heavy = np.array([VEHICLE_CLASSES.get(int(c), "") in ("bus", "truck") for c in classes])

    # Average over every frame the clip covers. A frame with no vehicle in the zone is a
    # real zero; skipping it would inflate density exactly when traffic is light.
    per_frame = len(frames) / covered
    pce_per_frame = equivalents.sum() / covered

    # The one external number in this module. `lane_count` comes from site.yaml, because
    # the interior lane lines are invisible at 320x240 and cannot be counted from the
    # image. Density is therefore per LANE-mile and scales as 1/lane_count, and so does
    # the Level of Service graded from it below. `vehicles_per_frame` above does not.
    lane_miles = (frame.zone_length_m / METRES_PER_MILE) * frame.lane_count
    density_veh = per_frame / lane_miles
    density_pce = pce_per_frame / lane_miles

    return DensityEstimate(
        vehicles_per_frame=float(per_frame),
        density_veh_per_mi_per_ln=float(density_veh),
        density_pc_per_mi_per_ln=float(density_pce),
        level_of_service=level_of_service.grade(density_pce),
        heavy_vehicle_fraction=float(heavy.mean()),
        n_frames=covered,
    )
