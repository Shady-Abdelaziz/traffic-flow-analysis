"""Vehicle count and classification.

Two things that sound like one. Counting asks how many distinct vehicles passed;
classification asks what they were.

**Counting is not "how many boxes did the detector draw".** That would count the same
car once per frame it appears in. Nor is it "how many track ids exist" -- a tracker that
loses a vehicle behind a truck and re-acquires it issues a second id for one car, so that
over-counts too, and a tracker that merges two vehicles under-counts.

The convention used here is **presence in the zone**: a vehicle is counted when its ground
point is inside the measurement zone in at least 5 frames. A counting line was used before
and under-counted badly: clips are 5.3 s, and a car queuing at 11 km/h never reaches a line
in the middle of the zone, so a whole heavy clip counted 2 vehicles. Requiring half a
second in the zone still drops detector flicker and short track fragments.

Classification takes a **majority vote across the track's frames**. At 320x240 a single
frame's class is noisy -- the same van reads as *car* then *truck* -- and voting over
thirty frames is markedly steadier than trusting whichever frame happened to be sampled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from trafficflow.geometry import RoadFrame
from trafficflow.tracks import VEHICLE_CLASSES, Track

__all__ = ["VehicleCount", "MIN_FRAMES_IN_ZONE", "count_vehicles"]


@dataclass(frozen=True, slots=True)
class VehicleCount:
    """Counts for one clip.

    Attributes
    ----------
    total:
        Distinct vehicles observed inside the measurement zone.
    by_class:
        Same total, split by vehicle type.
    per_lane:
        Counted vehicles by lane, which is where lane utilisation comes from.
    tracks_seen:
        Tracks that existed at all, counted or not. The gap between this and
        ``total`` is the honest measure of how many tracks were fragments.
    """

    total: int
    by_class: dict[str, int]
    per_lane: dict[int, int]
    tracks_seen: int

    @property
    def heavy_vehicle_percent(self) -> float:
        """Share of buses and trucks.

        Freight share is a genuine finding in its own right, and it is what drives the
        passenger-car-equivalent correction in the density estimate.
        """
        if not self.total:
            return 0.0
        heavy = self.by_class.get("bus", 0) + self.by_class.get("truck", 0)
        return 100.0 * heavy / self.total


#: Frames a track must spend inside the zone to count. Half a second: long enough to drop
#: detector flicker and short fragments, short enough that a queued car still counts.
MIN_FRAMES_IN_ZONE = 5


def count_vehicles(
    tracks: list[Track], frame: RoadFrame, min_frames_in_zone: int = MIN_FRAMES_IN_ZONE
) -> VehicleCount:
    """Count vehicles observed inside the measurement zone, by type and by lane.

    The lane is the one the vehicle spent most of its in-zone frames in.
    """
    by_class: Counter[str] = Counter()
    per_lane: Counter[int] = Counter()
    total = 0

    for track in tracks:
        road = frame.project_boxes(track.boxes)
        inside = frame.in_zone(road)
        if inside.sum() < min_frames_in_zone:
            continue
        total += 1
        by_class[track.vehicle_class] += 1
        lanes, counts = np.unique(frame.lane_index(road[inside, 0]), return_counts=True)
        per_lane[int(lanes[counts.argmax()])] += 1

    return VehicleCount(
        total=total,
        by_class={name: by_class.get(name, 0) for name in VEHICLE_CLASSES.values()},
        per_lane={lane: per_lane.get(lane, 0) for lane in range(1, frame.lane_count + 1)},
        tracks_seen=len(tracks),
    )
