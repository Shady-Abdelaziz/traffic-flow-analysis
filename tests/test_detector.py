"""The live page and the batch pass use one tracker, fed one frame at a time."""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from trafficflow.config import Paths, load_site  # noqa: E402
from trafficflow.detector import DetectorConfig, FrameTracker  # noqa: E402
from trafficflow.video import iter_frames  # noqa: E402

CLIP = Paths().archive / "video" / "cctv052x2004080516x01640.avi"

@pytest.mark.slow
@pytest.mark.skipif(not CLIP.exists(), reason="needs the dataset")
def test_ids_persist_across_frames():
    image = load_site().image
    tracker = FrameTracker(DetectorConfig(imgsz=640))
    rows = [row for n, frame in iter_frames(CLIP, max_frames=20) for row in tracker.track(frame, n, image)]
    assert rows, "vehicles should be found"
    ids = [row[1] for row in rows]
    assert max(ids.count(i) for i in set(ids)) >= 10, "one vehicle should keep its id"
    assert all(row[4] + row[6] >= 2 * image.roadway_crop_x for row in rows), "northbound dropped"


def test_search_range_extends_past_the_chosen_input_size():
    """The sweep must be able to return something larger than what is configured.

    ``imgsz`` was 960 while :func:`compare_input_sizes` only offered ``(320, 640, 960)``,
    so the measurement that justified 960 could not have returned anything better -- the
    answer was the top of the search range, not an optimum. Whatever the default becomes,
    the sweep has to reach past it or the same artifact returns.
    """
    from inspect import signature

    from trafficflow.detector import compare_input_sizes

    sizes = signature(compare_input_sizes).parameters["sizes"].default
    assert max(sizes) > DetectorConfig().imgsz, (
        f"sweep tops out at {max(sizes)}, which is the configured imgsz "
        f"{DetectorConfig().imgsz}: a better size cannot be discovered"
    )


def test_confidence_does_not_starve_bytetracks_second_stage():
    """The model threshold must sit at or below the tracker's low band.

    ByteTrack matches strong boxes first, then uses the weak band between
    ``track_low_thresh`` and ``track_high_thresh`` to recover tracks that would otherwise
    be dropped. The detector was passing 0.25 -- the tracker's *high* threshold -- so the
    weak band was deleted before the tracker saw it and the second stage ran on nothing.
    Plainly visible cars blinked out and back as their score crossed 0.25.
    """
    import yaml
    from ultralytics.utils import ROOT

    config = DetectorConfig()
    tracker = yaml.safe_load((ROOT / "cfg" / "trackers" / config.tracker).read_text())
    assert config.confidence <= tracker["track_low_thresh"], (
        f"conf={config.confidence} is above track_low_thresh="
        f"{tracker['track_low_thresh']}: ByteTrack's recovery stage gets nothing"
    )

