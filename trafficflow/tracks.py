"""The track table: what detection writes and every later stage reads.

Detection is the only expensive step in the project, so it runs once and its result is
written to disk. Everything after it -- speeds, density, lane occupancy, trajectories,
charts, the annotated video -- reads these tables back in milliseconds. Changing a
threshold or re-rendering a figure never re-runs the detector.

One CSV per clip, one row per detected vehicle per frame:

    frame, track_id, cls, conf, x1, y1, x2, y2

``track_id`` is the value that makes speed measurable at all. A detector alone sees each
frame afresh and cannot tell that the car in frame 5 is the car in frame 6; the tracker
assigns an identity that persists, and a speed is the change in one identity's position
over time.

CSV rather than a binary format on purpose: the whole database is about 130k rows, the
files open in any text editor or spreadsheet, and a grader can look at one without
installing anything.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "COLUMNS", "VEHICLE_CLASSES", "Track", "TrackTable", "read_settings",
    "settings_path", "stale_fields", "write_settings", "write_tracks",
]


#: Column order of the on-disk table. Fixed, because the files outlive any one run.
COLUMNS = ("frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2")

#: COCO class ids the detector is restricted to, and their names: every road vehicle COCO
#: has (its other vehicles are airplane, train and boat). COCO already distinguishes them,
#: which is why the detector needs no fine-tuning (and could not be fine-tuned -- the
#: database ships no bounding boxes). Bicycles are banned on I-5, so one detected here is
#: most likely a misread motorcycle; it is kept so that shows up rather than vanishing.
VEHICLE_CLASSES: dict[int, str] = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True, slots=True)
class Track:
    """One vehicle's path through a clip.

    Attributes
    ----------
    track_id:
        Identity assigned by the tracker, unique within the clip.
    frames:
        Frame numbers, ascending. May have gaps where the tracker lost and re-acquired
        the vehicle.
    boxes:
        ``(N, 4)`` of ``x1, y1, x2, y2`` in pixels, aligned with ``frames``.
    classes:
        COCO class id per frame. A tracker can flip between *car* and *truck* on the
        same vehicle, which is why :attr:`vehicle_class` votes rather than taking the
        first value.
    confidences:
        Detector confidence per frame.
    """

    track_id: int
    frames: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    confidences: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def vehicle_class(self) -> str:
        """Majority vote over every frame of the track.

        A single frame's classification is noisy at 320x240 -- a van reads as *car* in
        one frame and *truck* in the next. Voting across the whole track is markedly
        more stable than trusting any one detection.
        """
        if len(self.classes) == 0:
            return "unknown"
        values, counts = np.unique(self.classes, return_counts=True)
        return VEHICLE_CLASSES.get(int(values[counts.argmax()]), "unknown")


class TrackTable:
    """All tracks of one clip, loaded from its CSV.

    Examples
    --------
    >>> table = TrackTable.load("output/tracks/cctv052x2004080516x01638.csv")  # doctest: +SKIP
    >>> len(table.tracks())                                                    # doctest: +SKIP
    17
    """

    def __init__(self, rows: np.ndarray, clip: str = "") -> None:
        self.rows = rows
        self.clip = clip

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "TrackTable":
        """Read one clip's table.

        An empty table is legitimate -- a clip may genuinely contain no detections --
        so it loads to an empty array rather than raising.
        """
        target = Path(path)
        with target.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{target} is missing column(s): {sorted(missing)}")
            data = [[float(row[column]) for column in COLUMNS] for row in reader]
        rows = np.asarray(data, dtype=np.float64).reshape(-1, len(COLUMNS))
        return cls(rows, clip=target.stem)

    def __len__(self) -> int:
        return len(self.rows)

    # -- columns --------------------------------------------------------------

    @property
    def frames(self) -> np.ndarray:
        return self.rows[:, 0].astype(int)

    @property
    def track_ids(self) -> np.ndarray:
        return self.rows[:, 1].astype(int)

    @property
    def classes(self) -> np.ndarray:
        return self.rows[:, 2].astype(int)

    @property
    def confidences(self) -> np.ndarray:
        return self.rows[:, 3]

    @property
    def boxes(self) -> np.ndarray:
        """``(N, 4)`` of ``x1, y1, x2, y2``."""
        return self.rows[:, 4:8]

    # -- views ----------------------------------------------------------------

    def tracks(self, min_frames: int = 1) -> list[Track]:
        """Group rows into per-vehicle tracks, sorted by frame within each.

        Parameters
        ----------
        min_frames:
            Drop tracks seen in fewer frames than this. Very short tracks are usually
            spurious, and a speed fitted to two or three points is dominated by
            detector jitter rather than motion.
        """
        result: list[Track] = []
        for track_id in np.unique(self.track_ids):
            mask = self.track_ids == track_id
            order = np.argsort(self.rows[mask, 0], kind="stable")
            subset = self.rows[mask][order]
            if len(subset) < min_frames:
                continue
            result.append(
                Track(
                    track_id=int(track_id),
                    frames=subset[:, 0].astype(int),
                    boxes=subset[:, 4:8],
                    classes=subset[:, 2].astype(int),
                    confidences=subset[:, 3],
                )
            )
        return result

    def frame_rows(self, frame: int) -> np.ndarray:
        """Every detection in one frame, as raw rows."""
        return self.rows[self.frames == frame]

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_number, rows)`` for each frame that has detections."""
        for frame in np.unique(self.frames):
            yield int(frame), self.frame_rows(int(frame))

    def vehicles_per_frame(self) -> dict[int, int]:
        """Detection count keyed by frame. The raw material of the density estimate."""
        frames, counts = np.unique(self.frames, return_counts=True)
        return dict(zip(frames.tolist(), counts.tolist()))


def write_tracks(path: Path | str, rows: list[tuple]) -> Path:
    """Write one clip's table.

    Parameters
    ----------
    rows:
        Tuples in :data:`COLUMNS` order.

    Returns
    -------
    pathlib.Path
        Where the table was written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and swapped in whole. Opening the target itself truncated
    # the saved table first, so a server stopped mid-write lost a result that had been fine.
    partial = target.with_name(target.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    int(row[0]), int(row[1]), int(row[2]),
                    f"{row[3]:.4f}",
                    *(f"{value:.2f}" for value in row[4:8]),
                ]
            )
    os.replace(partial, target)
    return target


def settings_path(table: Path | str) -> Path:
    """Where one table's detector settings live: beside it, same stem, ``.json``."""
    table = Path(table)
    return table.with_suffix(".json")


def write_settings(table: Path | str, config) -> Path:
    """Record the detector settings that produced ``table``.

    Kept per table rather than per directory so that re-detecting a single clip -- which
    the live page does whenever it plays one -- cannot make the other 253 tables look
    current. Each table carries its own provenance or none at all.

    Without this a table is just numbers: change ``imgsz`` or ``confidence`` and every
    saved table is silently wrong, while the live page keeps reusing them and the batch
    pass keeps skipping the clips that have them. Nothing anywhere notices.
    """
    path = settings_path(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = asdict(config) if is_dataclass(config) else dict(config)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)
    return path


def read_settings(table: Path | str) -> dict | None:
    """The settings recorded for one table, or None if there are none to read."""
    path = settings_path(table)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None            # half-written, from a pass that died mid-record


def stale_fields(table: Path | str, config) -> tuple[str, ...]:
    """Which settings ``table`` disagrees with ``config`` about.

    Empty when they agree, and empty when the table does not exist -- nothing absent is
    stale. A table with no recorded settings is stale in every field: its provenance is
    unknown, and assuming it matches is the failure this exists to prevent.
    """
    if not Path(table).exists():
        return ()
    wanted = asdict(config) if is_dataclass(config) else dict(config)
    saved = read_settings(table)
    if saved is None:
        return tuple(sorted(wanted))
    return tuple(sorted(key for key, value in wanted.items() if saved.get(key) != value))
