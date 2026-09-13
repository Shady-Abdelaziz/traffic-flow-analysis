"""The database index files agree with each other, and folds resolve to the right clips.

The failure this guards against is quiet. ``info.txt`` and ``ImageMaster`` describe the
same 254 clips in *different orders*, and the published folds index into the second one.
Using a fold index against the first returns a different set of clips that still carries
plausible labels, so nothing raises and nothing looks wrong -- the accuracy number is
simply meaningless. These tests make that mismatch loud.
"""

from __future__ import annotations

import pytest

from trafficflow.config import Paths
from trafficflow.dataset import TRAFFIC_CLASSES, Clip, TrafficDatabase

EXPECTED_CLIPS = 254
EXPECTED_CLASS_COUNTS = {"light": 165, "medium": 45, "heavy": 44}


@pytest.fixture(scope="module")
def db() -> TrafficDatabase:
    return TrafficDatabase.load(Paths().archive)


def test_loads_every_clip(db: TrafficDatabase) -> None:
    assert len(db) == EXPECTED_CLIPS
    assert db.class_counts() == EXPECTED_CLASS_COUNTS


def test_master_order_differs_from_info_order(db: TrafficDatabase) -> None:
    """The premise behind every other test here.

    If these two orders ever coincided, joining by index would start working by
    accident and the bug this module exists to prevent would stop being detectable.
    """
    info_order = [clip.name for clip in db.clips]
    assert info_order != list(db.master_order)
    assert set(info_order) == set(db.master_order)


def test_master_order_is_grouped_by_class(db: TrafficDatabase) -> None:
    """ImageMaster runs heavy, then medium, then light -- which is why it differs."""
    labels = [db[name].traffic_class for name in db.master_order]
    assert labels[0] == "heavy"
    assert labels[-1] == "light"
    # each class occupies one contiguous run
    runs = [label for i, label in enumerate(labels) if i == 0 or label != labels[i - 1]]
    assert runs == ["heavy", "medium", "light"]


def test_folds_are_complements(db: TrafficDatabase) -> None:
    assert len(db.folds) == 4
    for fold in db.folds:
        assert not set(fold.train) & set(fold.test)
        assert len(fold.train) + len(fold.test) == EXPECTED_CLIPS


def test_every_clip_is_tested_exactly_once(db: TrafficDatabase) -> None:
    """Pooling predictions across folds is only fair if this holds."""
    tested = [name for fold in db.folds for name in fold.test]
    assert len(tested) == EXPECTED_CLIPS
    assert len(set(tested)) == EXPECTED_CLIPS


def test_folds_hold_names_not_indices(db: TrafficDatabase) -> None:
    """Folds must arrive already resolved, so callers cannot index the wrong table."""
    for fold in db.folds:
        for name in (*fold.train, *fold.test):
            assert isinstance(name, str)
            assert name in db.labels()


def test_folds_keep_every_class_in_training(db: TrafficDatabase) -> None:
    """A fold missing a class would make class weights divide by zero."""
    for fold in db.folds:
        present = {db[name].traffic_class for name in fold.train}
        assert present == set(TRAFFIC_CLASSES)


def test_hour_parsed_from_name_matches_timestamp_column(db: TrafficDatabase) -> None:
    """Two independent encodings of the recording hour, cross-checked."""
    for clip in db.clips:
        assert clip.hour == int(clip.timestamp.split(".")[0])
        assert 0 <= clip.hour <= 23


def test_corrupted_first_frame_is_recorded(db: TrafficDatabase) -> None:
    """Every clip declares that frame 1 is unusable; the reader must honour it."""
    assert {clip.start_frame for clip in db.clips} == {2}
    assert all(clip.usable_frame_count == clip.n_frames - 1 for clip in db.clips)


def test_constant_columns_carry_no_information(db: TrafficDatabase) -> None:
    """Direction and day/night never vary, so they must never become model features."""
    assert {clip.direction for clip in db.clips} == {"south"}
    assert {clip.day_night for clip in db.clips} == {"day"}


def test_day_index_splits_the_two_recording_days(db: TrafficDatabase) -> None:
    """The camera moved between days, so the split has to be recoverable."""
    days = {clip.day_index for clip in db.clips}
    assert days == {1, 2}
    assert {clip.date for clip in db.clips if clip.day_index == 1} == {"20040805"}


def test_majority_baseline_matches_the_published_figure(db: TrafficDatabase) -> None:
    assert db.majority_baseline() == pytest.approx(165 / 254, abs=1e-4)


def test_video_files_all_exist(db: TrafficDatabase) -> None:
    missing = [clip.name for clip in db.clips if not clip.video_path(db.data_root).exists()]
    assert not missing, f"{len(missing)} clips have no video file, first {missing[:3]}"


def test_row_of_wrong_width_is_rejected() -> None:
    """A truncated index file should fail loudly at load, not silently mis-parse."""
    with pytest.raises(ValueError, match="expected 10"):
        Clip.from_row(["cctv052x2004080516x01638", "20040805", "16.01638"])
