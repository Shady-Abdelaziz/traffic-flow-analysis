"""The split the model is scored on, and the leak the shipped one carries.

The archive's folds are a round-robin by position while consecutive clips are
consecutive recordings, so a clip's five-seconds-later near-duplicate sits in its own
training fold. These tests pin the size of that problem and the fix, because both are
claims the report makes out loud.

They lived at the foot of ``test_dataset.py``, which is about whether the index files
agree with each other. That module answers "did we read the database correctly"; this
one answers "is the number we report from it honest", and they fail for different
reasons.
"""

from __future__ import annotations

import pytest

from trafficflow.config import Paths
from trafficflow.dataset import TRAFFIC_CLASSES, TrafficDatabase
from trafficflow.evaluate import (
    grouped_folds,
    nearest_recording_baseline,
    recording_runs,
)


@pytest.fixture(scope="module")
def db() -> TrafficDatabase:
    return TrafficDatabase.load(Paths().archive)


def test_the_clips_form_fourteen_separate_recordings(db: TrafficDatabase) -> None:
    """If they were one unbroken run, no split could keep near-duplicates together."""
    runs = recording_runs(db)
    assert len(set(runs.values())) == 14
    assert len(runs) == len(db)


def test_the_shipped_folds_split_adjacent_recordings(db: TrafficDatabase) -> None:
    """98.4% of test clips have a neighbouring recording in their training fold."""
    touching = sum(
        1
        for fold in db.folds
        for clip in fold.test
        if {(db[clip].date, db[clip].serial - 1), (db[clip].date, db[clip].serial + 1)}
        & {(db[other].date, db[other].serial) for other in fold.train}
    )
    assert touching / len(db) > 0.95


def test_grouped_folds_never_split_a_recording(db: TrafficDatabase) -> None:
    """The fix, checked rather than assumed -- and the classes still spread."""
    folds = grouped_folds(db)
    assert len(folds) == 4
    assert sorted(clip for fold in folds for clip in fold.test) == sorted(
        clip.name for clip in db
    )

    for fold in folds:
        train = {(db[clip].date, db[clip].serial) for clip in fold.train}
        for clip in fold.test:
            date, serial = db[clip].date, db[clip].serial
            assert not {(date, serial - 1), (date, serial + 1)} & train, (
                f"{clip} has an adjacent recording in its training fold"
            )
        counts = [
            sum(1 for clip in fold.test if db[clip].traffic_class == label)
            for label in TRAFFIC_CLASSES
        ]
        assert min(counts) >= 7, f"fold {fold.index} class counts {counts}"


def test_reading_no_pixels_scores_far_higher_on_the_shipped_folds(
    db: TrafficDatabase,
) -> None:
    """The leak, measured. This is why the grouped split is the default."""
    shipped = nearest_recording_baseline(db, folds=db.folds)
    grouped = nearest_recording_baseline(db, folds=grouped_folds(db))

    assert shipped.macro_f1 == pytest.approx(0.790, abs=0.01)
    assert grouped.macro_f1 == pytest.approx(0.584, abs=0.01)
    assert shipped.macro_f1 - grouped.macro_f1 > 0.15
