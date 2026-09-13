"""The split the model is scored on.

Consecutive clips are consecutive recordings, so the folds keep each recording run
whole. These tests pin that, and the no-pixel bar a model has to clear on it.

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


def test_grouped_folds_never_split_a_recording(db: TrafficDatabase) -> None:
    """Checked rather than assumed -- and the classes still spread."""
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


def test_reading_no_pixels_sets_the_bar(db: TrafficDatabase) -> None:
    """The score available from recording order alone, on the grouped folds."""
    assert nearest_recording_baseline(db).macro_f1 == pytest.approx(0.584, abs=0.01)
