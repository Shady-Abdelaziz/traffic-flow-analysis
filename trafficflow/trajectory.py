"""Vehicle trajectories: bird's-eye paths and the time-space diagram.

The exam names "turning, merging" as example trajectory analyses. Both deserve a direct
answer rather than a fabricated metric:

**Turning does not occur here.** This is a freeway mainline. Vehicles travel in one
direction and leave only at ramps. Reporting a turn count would be reporting noise.

**Merging is not measurable from this camera.** No junction appears in frame: the edge
markings run unbroken to the far end of the visible road, so any ramp lies beyond it, at a
range where a vehicle is a handful of pixels anyway. The code below detects merges where
they are visible, and the honest expected result on this footage is close to zero events.

What *is* measurable, and more useful, is the **time-space diagram**: every vehicle's
distance plotted against time, one line each. It is the standard tool of traffic flow
theory, and it makes visible what no single number shows -- the slope of each line is that
vehicle's speed, so a shockwave travelling back through traffic appears directly as a bend
propagating across the lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficflow.geometry import RoadFrame
from trafficflow.tracks import Track

__all__ = [
    "Trajectory",
    "build_trajectories",
    "detect_merges",
    "TURNING_NOTE",
    "MERGING_NOTE",
]


#: Reported verbatim rather than silently omitted -- a parameter that cannot apply is a
#: finding, not a gap.
TURNING_NOTE = (
    "Not applicable. This is an Interstate mainline: traffic is one-way and there are "
    "no turning movements to measure. Vehicles leave only at ramps."
)

MERGING_NOTE = (
    "Not measurable from this viewpoint. No junction enters the frame -- the edge "
    "markings run unbroken to the far end of the visible road -- so any merge happens "
    "beyond the measurement zone, at a range where vehicles are only a few pixels "
    "across. Merge detection is implemented and run; a near-zero count is the expected "
    "and honest result."
)


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One vehicle's path in road coordinates.

    Attributes
    ----------
    times_s:
        Seconds from the start of the clip.
    lateral_m:
        Metres right of the yellow left-edge line.
    distance_m:
        Metres from the camera along the road.
    """

    track_id: int
    vehicle_class: str
    times_s: np.ndarray
    lateral_m: np.ndarray
    distance_m: np.ndarray

    def __len__(self) -> int:
        return len(self.times_s)

    @property
    def lateral_drift_m(self) -> float:
        """Total lateral movement, the raw signal a lane change shows up in."""
        return float(self.lateral_m.max() - self.lateral_m.min()) if len(self) else 0.0

    @property
    def distance_covered_m(self) -> float:
        return float(abs(self.distance_m[-1] - self.distance_m[0])) if len(self) else 0.0


def build_trajectories(
    tracks: list[Track], frame: RoadFrame, fps: float, min_points: int = 5
) -> list[Trajectory]:
    """Project tracks onto the road plane, keeping only points inside the zone."""
    result: list[Trajectory] = []
    for track in tracks:
        road = frame.project_boxes(track.boxes)
        inside = frame.in_zone(road)
        if inside.sum() < min_points:
            continue
        result.append(
            Trajectory(
                track_id=track.track_id,
                vehicle_class=track.vehicle_class,
                times_s=track.frames[inside] / fps,
                lateral_m=road[inside, 0],
                distance_m=road[inside, 1],
            )
        )
    return result


def detect_merges(
    trajectories: list[Trajectory],
    frame: RoadFrame,
    min_lateral_m: float = 2.0,
) -> list[dict]:
    """Find vehicles entering the roadway from the right edge.

    A merge shows as a vehicle that starts in or beyond the rightmost lane and moves
    left by at least a lane width while travelling. See :data:`MERGING_NOTE` for why
    this is expected to find little on this footage -- the result is reported as
    measured either way, rather than being quietly dropped.
    """
    rightmost_boundary = frame.travel_width_m - frame.lane_width_m
    merges: list[dict] = []
    for trajectory in trajectories:
        if len(trajectory) < 5:
            continue
        started_right = trajectory.lateral_m[0] >= rightmost_boundary
        moved_left = trajectory.lateral_m[0] - trajectory.lateral_m[-1] >= min_lateral_m
        if started_right and moved_left:
            merges.append(
                {
                    "track_id": trajectory.track_id,
                    "from_lateral_m": float(trajectory.lateral_m[0]),
                    "to_lateral_m": float(trajectory.lateral_m[-1]),
                    "at_distance_m": float(trajectory.distance_m[0]),
                }
            )
    return merges


def summarise(trajectories: list[Trajectory], merges: list[dict]) -> dict:
    """Clip-level trajectory statistics, with the two inapplicable cases stated."""
    return {
        "n_trajectories": len(trajectories),
        "mean_distance_covered_m": (
            float(np.mean([t.distance_covered_m for t in trajectories])) if trajectories else 0.0
        ),
        "mean_lateral_drift_m": (
            float(np.mean([t.lateral_drift_m for t in trajectories])) if trajectories else 0.0
        ),
        "n_merges": len(merges),
        "turning": TURNING_NOTE,
        "merging": MERGING_NOTE,
    }
