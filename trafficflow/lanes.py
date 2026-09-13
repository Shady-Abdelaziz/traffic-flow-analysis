"""Lane occupancy and lane changes.

**Occupancy** is the fraction of time a lane has a vehicle in it. It is worth
recognising what this reproduces: the dataset's own README records ``Loop: 152.2``, an
inductive loop detector buried in the pavement at that milepost. A loop measures exactly
one thing -- the fraction of time metal sits above it -- and that is occupancy. So this
computes, from video, the quantity the road itself was already instrumented to measure.

**Lane changes** need care. A vehicle's lateral position wobbles by a fraction of a lane
from frame to frame, because the detection box breathes and its bottom-centre moves with
it. A vehicle travelling straight along a lane boundary will therefore appear to cross it
repeatedly. Counting every crossing would report dozens of lane changes for a vehicle
that never changed lane.

Two rules suppress that. A change is only recorded once the vehicle has settled in the
new lane for :data:`DEBOUNCE_FRAMES` consecutive frames, and the crossing must clear the
boundary by :data:`HYSTERESIS_M` rather than merely touch it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficflow.geometry import RoadFrame
from trafficflow.tracks import Track, TrackTable

__all__ = [
    "DEBOUNCE_FRAMES",
    "HYSTERESIS_M",
    "LaneChange",
    "LaneOccupancy",
    "lane_occupancy",
    "detect_lane_changes",
    "summarise_changes",
]


#: Consecutive frames a vehicle must hold a new lane before the move counts. Five frames
#: is half a second at 10 fps -- longer than any jitter, far shorter than a real change.
DEBOUNCE_FRAMES = 5

#: Distance past a lane boundary before the vehicle is considered to have crossed it.
#: 0.4 m is roughly one pixel of lateral error mapped to the road at mid-zone.
HYSTERESIS_M = 0.4


@dataclass(frozen=True, slots=True)
class LaneOccupancy:
    """Per-lane occupancy over one clip.

    Attributes
    ----------
    occupancy:
        Lane number to the fraction of frames in which that lane held at least one
        vehicle. This is the video analogue of an inductive loop's output.
    vehicle_counts:
        Lane number to the number of distinct vehicles seen in it.
    """

    occupancy: dict[int, float]
    vehicle_counts: dict[int, int]
    n_frames: int

    @property
    def busiest_lane(self) -> int:
        """Lane with the highest occupancy, 0 if nothing was seen."""
        return max(self.occupancy, key=self.occupancy.get, default=0)

    @property
    def imbalance(self) -> float:
        """Spread between the busiest and quietest lane.

        Near zero means traffic is using the road evenly. A large value means it is
        not, which on a freeway usually points at a downstream bottleneck or at
        vehicles avoiding a lane -- both worth reporting.
        """
        if not self.occupancy:
            return 0.0
        values = list(self.occupancy.values())
        return float(max(values) - min(values))


@dataclass(frozen=True, slots=True)
class LaneChange:
    """One confirmed lane change."""

    track_id: int
    frame: int
    from_lane: int
    to_lane: int

    @property
    def direction(self) -> str:
        """``left`` or ``right``, as seen from the driver."""
        return "left" if self.to_lane < self.from_lane else "right"


def lane_occupancy(
    table: TrackTable, frame: RoadFrame, n_frames: int | None = None
) -> LaneOccupancy:
    """Fraction of frames each lane was occupied, within the measurement zone.

    Parameters
    ----------
    n_frames:
        How many frames the clip actually contains. Pass it whenever it is known.
        Without it the denominator can only be inferred from the frames that happen
        to carry a detection, and in light traffic -- where most frames are empty --
        that denominator is a fraction of the clip, which pushes every lane's
        occupancy toward 1.0 and reports an empty road as a busy one.
        :func:`trafficflow.density.estimate_density` takes it for the same reason.
    """
    lanes = range(1, frame.lane_count + 1)
    empty = LaneOccupancy({lane: 0.0 for lane in lanes}, {lane: 0 for lane in lanes},
                          n_frames or 0)
    if len(table) == 0:
        return empty

    road = frame.project_boxes(table.boxes)
    inside = frame.in_zone(road)
    if not inside.any():
        return empty

    lane_index = frame.lane_index(road[inside, 0])
    frames = table.frames[inside]
    ids = table.track_ids[inside]
    total_frames = n_frames or int(table.frames.max() - table.frames.min() + 1)

    occupancy: dict[int, float] = {}
    counts: dict[int, int] = {}
    for lane in lanes:
        mask = lane_index == lane
        occupancy[lane] = float(len(np.unique(frames[mask])) / total_frames) if total_frames else 0.0
        counts[lane] = int(len(np.unique(ids[mask])))
    return LaneOccupancy(occupancy, counts, total_frames)


def detect_lane_changes(track: Track, frame: RoadFrame) -> list[LaneChange]:
    """Find confirmed lane changes in one track.

    A change is recorded only when the vehicle crosses a boundary by more than
    :data:`HYSTERESIS_M` and then holds the new lane for :data:`DEBOUNCE_FRAMES`
    consecutive frames. Without both rules, a vehicle tracking straight along a painted
    line produces a stream of false changes as its box jitters across the boundary.
    """
    road = frame.project_boxes(track.boxes)
    inside = frame.in_zone(road)
    if inside.sum() < DEBOUNCE_FRAMES + 1:
        return []

    x = road[inside, 0]
    frames = track.frames[inside]
    width = frame.lane_width_m

    # Assign a lane only when the position is clear of the boundaries on both sides;
    # otherwise carry the previous lane forward, so straddling is not a change.
    lanes = np.floor(x / width).astype(int) + 1
    offset_in_lane = x - (lanes - 1) * width
    ambiguous = (offset_in_lane < HYSTERESIS_M) | (offset_in_lane > width - HYSTERESIS_M)

    settled = lanes.copy()
    for i in range(len(settled)):
        if ambiguous[i]:
            settled[i] = settled[i - 1] if i else lanes[i]

    changes: list[LaneChange] = []
    current = settled[0]
    for i in range(1, len(settled)):
        if settled[i] == current:
            continue
        # confirm: the new lane must hold for DEBOUNCE_FRAMES consecutive samples
        window = settled[i : i + DEBOUNCE_FRAMES]
        if len(window) >= DEBOUNCE_FRAMES and np.all(window == settled[i]):
            changes.append(
                LaneChange(
                    track_id=track.track_id,
                    frame=int(frames[i]),
                    from_lane=int(current),
                    to_lane=int(settled[i]),
                )
            )
            current = settled[i]
    return changes


def summarise_changes(changes: list[LaneChange], n_vehicles: int) -> dict[str, float]:
    """Aggregate lane changes across a clip.

    The rate per vehicle is the comparable figure: a clip with more traffic will show
    more changes simply because it has more vehicles, and dividing removes that.
    """
    return {
        "n_lane_changes": len(changes),
        "changes_per_vehicle": len(changes) / n_vehicles if n_vehicles else 0.0,
        "changes_left": sum(1 for change in changes if change.direction == "left"),
        "changes_right": sum(1 for change in changes if change.direction == "right"),
    }
