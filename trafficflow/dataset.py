"""Access to the UCSD/WSDOT I-5 traffic video database.

The database ships three index files that describe the same 254 clips in different
orders, which is the single easiest thing to get wrong when using it:

``info.txt``
    One tab-separated row per clip holding the recording metadata. Several of its
    columns are constant across the entire database (see :data:`CONSTANT_COLUMNS`)
    and carry no information.
``ImageMaster``
    The clip order used by the *published* evaluation protocol. It is **not** the
    order of ``info.txt`` -- it is grouped by traffic class: heavy first, then
    medium, then light.
``EvalSet_train`` / ``EvalSet_test``
    Four train/test folds, stored as 0-based indices into ``ImageMaster`` order.

Hence the rule this module exists to enforce: **join on filename, never on row
index.** Indexing ``info.txt`` with a fold index silently yields the wrong clips --
the labels still look plausible, so the mistake survives until the accuracy numbers
stop making sense.

:meth:`TrafficDatabase.load` cross-checks the files against each other and raises on
any disagreement, so a damaged or partial copy of the database fails loudly at
startup rather than halfway through an hour-long detection run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Clip", "Fold", "TrafficDatabase", "TRAFFIC_CLASSES", "CONSTANT_COLUMNS"]


#: Traffic classes, ordered from free-flowing to stopped. The database README defines
#: them as free-flowing / reduced speed / stopped-or-very-slow respectively.
TRAFFIC_CLASSES: tuple[str, ...] = ("light", "medium", "heavy")

#: Columns of ``info.txt`` that never vary across the 254 clips. They are parsed and
#: validated, but carry no information and must not be used as model features.
CONSTANT_COLUMNS: dict[str, str] = {
    "direction": "south",
    "day_night": "day",
    "start_frame": "2",
}

#: Filename layout: ``cctv052x<yyyymmdd><HH>x<serial>``. For example
#: ``cctv052x2004080516x01638`` decomposes into date 2004-08-05, hour 16, serial 01638.
_NAME_RE = re.compile(r"^cctv052x(?P<date>\d{8})(?P<hour>\d{2})x(?P<serial>\d+)$")

_EXPECTED_CLIP_COUNT = 254
_INFO_COLUMNS = 10


def _read_table(path: Path) -> list[list[str]]:
    """Read a tab-separated index file, dropping its leading ``#`` comment header.

    Parameters
    ----------
    path:
        File to read. Encoding errors are replaced rather than raised: the database
        is plain ASCII, so any replacement means a damaged copy, which the
        structural checks in :meth:`TrafficDatabase.load` will then catch.

    Returns
    -------
    list of list of str
        One list of fields per data row.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [
            line.rstrip("\n").split("\t")
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


@dataclass(frozen=True, slots=True)
class Clip:
    """One video clip and its recording metadata.

    Attributes
    ----------
    name:
        Filename stem, e.g. ``cctv052x2004080516x01638``. This is the join key for
        every other index file.
    date:
        Recording date as ``YYYYMMDD``.
    timestamp:
        Raw ``info.txt`` timestamp, formatted ``<hour>.<serial>``. Despite the
        decimal point this is not a clock time, so :attr:`hour` is derived from the
        filename instead and the two are cross-checked on load.
    weather:
        One of ``overcast``, ``clear``, ``rain``.
    start_frame:
        First usable frame, 1-based. Always 2: the README states the first frame of
        every clip is corrupted by another video signal.
    n_frames:
        Total frames in the file, including the corrupted first one.
    traffic_class:
        Manual label, one of :data:`TRAFFIC_CLASSES`.
    notes:
        Free-text annotation. Empty for 240 of the 254 clips, otherwise
        ``drops on lens`` or ``ambiguous``.
    """

    name: str
    date: str
    timestamp: str
    direction: str
    day_night: str
    weather: str
    start_frame: int
    n_frames: int
    traffic_class: str
    notes: str

    @classmethod
    def from_row(cls, row: list[str]) -> "Clip":
        """Build a clip from one ``info.txt`` row, validating its width."""
        if len(row) != _INFO_COLUMNS:
            raise ValueError(
                f"info.txt row has {len(row)} fields, expected {_INFO_COLUMNS}: {row!r}"
            )
        return cls(
            name=row[0],
            date=row[1],
            timestamp=row[2],
            direction=row[3],
            day_night=row[4],
            weather=row[5],
            start_frame=int(row[6]),
            n_frames=int(row[7]),
            traffic_class=row[8],
            notes=row[9],
        )

    @property
    def hour(self) -> int:
        """Hour of day the clip was recorded, 0-23, parsed from the filename.

        Traffic class is strongly predictable from this alone -- hours 06-12 and
        19-20 are 100% light -- so it is a confound to be aware of, never a model
        feature and never a baseline worth shipping.
        """
        match = _NAME_RE.match(self.name)
        if match is None:  # pragma: no cover - load() rejects such names first
            raise ValueError(f"clip name does not match the expected layout: {self.name}")
        return int(match.group("hour"))

    @property
    def serial(self) -> int:
        """Recording order within the day, parsed from the filename.

        Consecutive serials are consecutive five-second recordings of the same
        traffic -- ``...x01638`` then ``...x01639`` -- so they are near-duplicates
        and must never be split across training and test. See
        :func:`trafficflow.evaluate.recording_runs`.
        """
        match = _NAME_RE.match(self.name)
        if match is None:  # pragma: no cover - load() rejects such names first
            raise ValueError(f"clip name does not match the expected layout: {self.name}")
        return int(match.group("serial"))

    @property
    def day_index(self) -> int:
        """1 for 2004-08-05, 2 for 2004-08-06.

        The camera shifted by a few pixels between the two days, so the calibration
        stores a per-day pixel offset rather than a second homography.
        """
        return 1 if self.date == "20040805" else 2

    @property
    def usable_frame_count(self) -> int:
        """Frames left after skipping the corrupted first frame."""
        return self.n_frames - (self.start_frame - 1)

    def video_path(self, data_root: Path | str) -> Path:
        """Path to this clip's ``.avi`` file inside ``data_root``."""
        return Path(data_root) / "video" / f"{self.name}.avi"


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split of the published four-fold evaluation protocol.

    Both members hold clip *names*, already resolved from the ``ImageMaster`` indices
    stored on disk, so callers never handle a raw index.
    """

    index: int
    train: tuple[str, ...]
    test: tuple[str, ...]


class TrafficDatabase:
    """The 254-clip database, with its index files cross-checked against each other.

    Examples
    --------
    >>> db = TrafficDatabase.load("archive")             # doctest: +SKIP
    >>> len(db)                                          # doctest: +SKIP
    254
    >>> db["cctv052x2004080516x01638"].traffic_class     # doctest: +SKIP
    'medium'
    """

    def __init__(
        self,
        data_root: Path,
        clips: tuple[Clip, ...],
        master_order: tuple[str, ...],
        folds: tuple[Fold, ...],
    ) -> None:
        self.data_root = data_root
        self.clips = clips
        self.master_order = master_order
        self.folds = folds
        self._by_name = {clip.name: clip for clip in clips}

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(cls, data_root: Path | str) -> "TrafficDatabase":
        """Parse and validate every index file under ``data_root``.

        Parameters
        ----------
        data_root:
            Directory holding ``info.txt``, ``ImageMaster``, ``EvalSet_train``,
            ``EvalSet_test`` and ``video/`` -- i.e. the unpacked database.

        Raises
        ------
        ValueError
            If the files disagree: mismatched clip sets, conflicting labels, folds
            that are not complements, or an unexpected clip count.
        """
        root = Path(data_root)
        clips = tuple(Clip.from_row(row) for row in _read_table(root / "info.txt"))
        master_rows = _read_table(root / "ImageMaster")
        master_order = tuple(row[1] for row in master_rows)
        master_labels = {row[1]: row[2] for row in master_rows}

        cls._check_clips(clips, master_order, master_labels)
        folds = cls._load_folds(root, master_order)
        return cls(root, clips, master_order, folds)

    @staticmethod
    def _check_clips(
        clips: tuple[Clip, ...],
        master_order: tuple[str, ...],
        master_labels: dict[str, str],
    ) -> None:
        """Validate the clip table against ``ImageMaster`` and the documented invariants."""
        if len(clips) != _EXPECTED_CLIP_COUNT:
            raise ValueError(
                f"info.txt lists {len(clips)} clips, expected {_EXPECTED_CLIP_COUNT}"
            )
        if len(master_order) != _EXPECTED_CLIP_COUNT:
            raise ValueError(
                f"ImageMaster lists {len(master_order)} clips, "
                f"expected {_EXPECTED_CLIP_COUNT}"
            )

        info_names = {clip.name for clip in clips}
        if info_names != set(master_order):
            missing = sorted(set(master_order) - info_names)
            extra = sorted(info_names - set(master_order))
            raise ValueError(
                f"info.txt and ImageMaster describe different clips; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )

        for clip in clips:
            if _NAME_RE.match(clip.name) is None:
                raise ValueError(f"clip name does not match the expected layout: {clip.name}")
            if clip.traffic_class not in TRAFFIC_CLASSES:
                raise ValueError(f"{clip.name}: unknown traffic class {clip.traffic_class!r}")
            if master_labels[clip.name] != clip.traffic_class:
                raise ValueError(
                    f"{clip.name}: info.txt says {clip.traffic_class!r} but "
                    f"ImageMaster says {master_labels[clip.name]!r}"
                )
            # The timestamp column encodes <hour>.<serial>. Agreeing with the hour
            # parsed out of the filename confirms both are being read correctly.
            if int(clip.timestamp.split(".")[0]) != clip.hour:
                raise ValueError(
                    f"{clip.name}: timestamp {clip.timestamp!r} disagrees with filename hour"
                )
            for field, expected in CONSTANT_COLUMNS.items():
                actual = str(getattr(clip, field))
                if actual != expected:
                    raise ValueError(
                        f"{clip.name}: {field} is {actual!r}, "
                        f"expected the constant {expected!r}"
                    )

    @staticmethod
    def _load_folds(root: Path, master_order: tuple[str, ...]) -> tuple[Fold, ...]:
        """Resolve the ``EvalSet`` index rows into folds of clip names."""

        def parse(filename: str) -> list[list[int]]:
            with (root / filename).open(encoding="utf-8") as handle:
                return [
                    [int(token) for token in line.strip().split(",")]
                    for line in handle
                    if line.strip()
                ]

        train_rows, test_rows = parse("EvalSet_train"), parse("EvalSet_test")
        if len(train_rows) != len(test_rows):
            raise ValueError(
                f"EvalSet_train has {len(train_rows)} folds but "
                f"EvalSet_test has {len(test_rows)}"
            )

        folds: list[Fold] = []
        for index, (train_idx, test_idx) in enumerate(zip(train_rows, test_rows)):
            if set(train_idx) & set(test_idx):
                raise ValueError(f"fold {index}: train and test overlap")
            if len(train_idx) + len(test_idx) != len(master_order):
                raise ValueError(
                    f"fold {index}: train+test covers {len(train_idx) + len(test_idx)} "
                    f"clips, expected {len(master_order)}"
                )
            folds.append(
                Fold(
                    index=index,
                    train=tuple(master_order[i] for i in train_idx),
                    test=tuple(master_order[i] for i in test_idx),
                )
            )
        return tuple(folds)

    # -- lookup ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        return iter(self.clips)

    def __getitem__(self, name: str) -> Clip:
        """Look up a clip by filename stem."""
        return self._by_name[name]

    def labels(self) -> dict[str, str]:
        """Map every clip name to its traffic class."""
        return {clip.name: clip.traffic_class for clip in self.clips}

    def by_class(self, traffic_class: str) -> list[Clip]:
        """All clips carrying ``traffic_class``, in ``info.txt`` order."""
        if traffic_class not in TRAFFIC_CLASSES:
            raise ValueError(f"unknown traffic class {traffic_class!r}")
        return [clip for clip in self.clips if clip.traffic_class == traffic_class]

    def class_counts(self) -> dict[str, int]:
        """Number of clips per traffic class, ordered light to heavy."""
        return {name: len(self.by_class(name)) for name in TRAFFIC_CLASSES}

    def majority_baseline(self) -> float:
        """Accuracy of always predicting the most common class (0.650).

        The floor, not the bar. A classifier must clear this *and* the much stronger
        no-pixel probe in :mod:`trafficflow.evaluate` -- which copies the label of the
        nearest recording -- before its score means anything.
        """
        return max(self.class_counts().values()) / len(self.clips)
