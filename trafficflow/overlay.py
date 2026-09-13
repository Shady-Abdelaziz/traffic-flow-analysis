"""Annotated video: the analysis drawn back onto the footage.

This is the deliverable a reader believes or doesn't. Numbers in a table can be wrong in
ways nobody notices; a box drawn on the wrong vehicle, or a car labelled 200 km/h, is
obvious in two seconds. So the overlay doubles as the project's most effective test.

What is drawn, and why each element is there:

- **The box and its track id**, so identity across frames can be checked by eye. If ids
  swap between vehicles, the speed estimates built on them are suspect.
- **Speed in km/h**, drawn only for tracks whose fit was reliable. Showing a number for
  an unreliable fit would imply a confidence the measurement does not have.
- **A level-of-service badge**, so the clip-level verdict sits next to the evidence for it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from trafficflow.dataset import Clip
from trafficflow.geometry import ground_point
from trafficflow.pipeline import ClipParameters
from trafficflow.tracks import VEHICLE_CLASSES, TrackTable
from trafficflow.video import iter_frames

__all__ = [
    "LOS_COLOURS", "annotate_frame", "clip_badge", "render_overlay",
]


#: Level of Service to colour, green through red. Matches the convention used on
#: traffic-management dashboards, so the meaning reads without a legend.
LOS_COLOURS = {
    "A": (110, 180, 90), "B": (140, 190, 90), "C": (90, 200, 220),
    "D": (70, 160, 240), "E": (70, 110, 240), "F": (60, 60, 220),
}

_CLASS_COLOURS = {
    "car": (200, 170, 90), "truck": (90, 140, 230),
    "bus": (150, 110, 220), "motorcycle": (110, 210, 150), "bicycle": (90, 210, 230),
    "unknown": (160, 160, 160),
}


def _draw_badge(canvas: np.ndarray, text: str, los: str) -> None:
    """A dark strip across the top: level-of-service letter, then one line of text."""
    colour = LOS_COLOURS.get(los, (160, 160, 160))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (28, 28, 28), -1)
    cv2.rectangle(canvas, (4, 5), (26, 25), colour, -1)
    cv2.putText(canvas, los, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (34, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)


def clip_badge(parameters: ClipParameters) -> tuple[str, str]:
    """Badge text and level of service for a whole analysed clip."""
    speed = parameters.speed_summary["median_kmh"]
    speed_text = f"{speed:.0f} km/h" if speed == speed else "no speed"   # NaN != NaN
    text = (
        f"{parameters.traffic_class.upper()}   {speed_text}   "
        f"{parameters.density.density_pc_per_mi_per_ln:.0f} pc/mi/ln   "
        f"{parameters.counts.total} veh"
    )
    return text, parameters.density.level_of_service


def _draw_label(canvas: np.ndarray, text: str, x: int, y: int, colour: tuple[int, int, int]) -> None:
    """White text on a dark tag in the vehicle's colour, so it reads over any background."""
    font, size, weight = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (width, height), baseline = cv2.getTextSize(text, font, size, weight)
    top = max(0, y - height - baseline - 6)
    x = min(max(0, x), max(0, canvas.shape[1] - width - 8))
    cv2.rectangle(canvas, (x, top), (x + width + 8, top + height + baseline + 6), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, top), (x + 3, top + height + baseline + 6), colour, -1)
    cv2.putText(canvas, text, (x + 6, top + height + 3), font, size, (255, 255, 255), weight, cv2.LINE_AA)


def annotate_frame(
    image: np.ndarray,
    rows: np.ndarray,
    speeds: dict,
    badge: tuple[str, str] | None = None,
    scale: int = 3,
) -> np.ndarray:
    """Draw every vehicle -- box, id, type, reliable speed -- and an optional badge onto one BGR frame.

    ``rows`` are this frame's track-table rows; ``speeds`` maps track id to its
    :class:`~trafficflow.speed.SpeedEstimate`. Nothing is drawn on the road surface itself,
    so the same drawing works for any camera.
    """
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    for row in rows:
        track_id, cls = int(row[1]), int(row[2])
        box = row[4:8] * scale
        name = VEHICLE_CLASSES.get(cls, "unknown")
        colour = _CLASS_COLOURS.get(name, (160, 160, 160))

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        label = f"#{track_id} {name}"
        estimate = speeds.get(track_id)
        if estimate is not None and estimate.is_reliable:
            label += f" {estimate.speed_kmh:.0f} km/h"
        _draw_label(canvas, label, x1, y1, colour)

        # the point the measurement actually uses -- the tyre contact patch
        gx, gy = ground_point(box)
        cv2.circle(canvas, (int(gx), int(gy)), 2, (255, 255, 255), -1, cv2.LINE_AA)

    if badge is not None:
        _draw_badge(canvas, *badge)
    return canvas


def render_overlay(
    clip: Clip,
    table: TrackTable,
    parameters: ClipParameters,
    data_root: Path | str,
    path: Path | str,
    *,
    scale: int = 3,
    fps: float | None = None,
) -> Path:
    """Write an annotated copy of one clip.

    Parameters
    ----------
    scale:
        Upscale factor. The source is 320x240; at that size the annotations are
        illegible, so the output is enlarged with nearest-neighbour interpolation --
        which keeps the original pixels visible rather than inventing smooth detail
        the footage never had.

    Returns
    -------
    pathlib.Path
        The written video.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows_by_frame = {number: rows for number, rows in table.iter_frames()}
    speeds = {estimate.track_id: estimate for estimate in parameters.speeds}
    badge = clip_badge(parameters)

    writer = None
    try:
        for number, image in iter_frames(clip.video_path(data_root)):
            rows = rows_by_frame.get(number, np.empty((0, 8)))
            canvas = annotate_frame(image, rows, speeds, badge, scale)
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps or 10.0, (width, height),
                )
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    return target
