"""Frame-level video access.

Every read of a clip goes through :func:`iter_frames` so that one database quirk is
handled in exactly one place: the README states that *the first frame of each clip is
corrupted with another video signal*, and ``info.txt`` repeats this as a
``start frame`` column that is 2 for all 254 clips. Decoding from frame 1 anywhere in
the pipeline injects a garbage frame that shows up later as a phantom detection, so no other module calls OpenCV directly.

The clips are uniform -- 320x240 at 10 fps, 52.9 frames on average -- which fixes two
constants used downstream: the time between consecutive frames is 0.1 s, and a
vehicle at the 60 mph speed limit advances roughly 2.7 m per frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "FIRST_USABLE_FRAME",
    "VideoInfo",
    "probe",
    "iter_frames",
    "read_frame",
]


#: First frame that is safe to decode, 1-based. The frame before it is corrupted by a
#: second video signal (database README, and the constant ``start frame`` column of
#: ``info.txt``).
FIRST_USABLE_FRAME = 2


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Container properties of one clip, as reported by the decoder."""

    width: int
    height: int
    fps: float
    frame_count: int


def probe(path: Path | str) -> VideoInfo:
    """Read container properties without decoding the whole clip.

    Raises
    ------
    FileNotFoundError
        If the decoder cannot open ``path``.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        return VideoInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_frames(
    path: Path | str,
    *,
    start_frame: int = FIRST_USABLE_FRAME,
    max_frames: int | None = None,
    step: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_number, image)`` pairs, skipping the corrupted lead-in frame.

    Parameters
    ----------
    path:
        Clip to decode.
    start_frame:
        First frame to yield, 1-based. Defaults to :data:`FIRST_USABLE_FRAME`; pass a
        larger value to skip further in, never a smaller one.
    max_frames:
        Stop after yielding this many frames. ``None`` reads to the end.
    step:
        Yield every ``step``-th frame. Used for sampling; note that frame numbers
        remain absolute, so downstream time arithmetic stays correct.

    Yields
    ------
    tuple of (int, numpy.ndarray)
        The 1-based frame number and its BGR image of shape ``(H, W, 3)``.

    Notes
    -----
    Frames are read sequentially and discarded up to ``start_frame`` rather than
    sought with ``CAP_PROP_POS_FRAMES``. These clips are short, and seeking in MJPEG
    AVI lands on the nearest key frame, which would silently shift the frame numbers
    that every speed estimate depends on.
    """
    if start_frame < FIRST_USABLE_FRAME:
        raise ValueError(
            f"start_frame={start_frame} would decode the corrupted lead-in frame; "
            f"the first usable frame is {FIRST_USABLE_FRAME}"
        )
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")

        emitted = 0
        frame_number = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number < start_frame:
                continue
            if (frame_number - start_frame) % step:
                continue
            yield frame_number, image
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        capture.release()


def read_frame(path: Path | str, frame_number: int = FIRST_USABLE_FRAME) -> np.ndarray:
    """Return a single frame by 1-based number.

    Raises
    ------
    IndexError
        If the clip ends before ``frame_number``.
    """
    for number, image in iter_frames(path, start_frame=frame_number, max_frames=1):
        if number == frame_number:
            return image
    raise IndexError(f"{path} has no frame {frame_number}")
