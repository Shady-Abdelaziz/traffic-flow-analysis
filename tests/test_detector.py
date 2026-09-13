"""The live page and the batch pass use one tracker, fed one frame at a time."""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from trafficflow.config import Paths, load_site  # noqa: E402
from trafficflow.detector import DetectorConfig, FrameTracker  # noqa: E402
from trafficflow.video import iter_frames  # noqa: E402

CLIP = Paths().archive / "video" / "cctv052x2004080516x01640.avi"
HEAVY_CLIP = Paths().archive / "video" / "cctv052x2004080516x01646.avi"


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


@pytest.mark.slow
@pytest.mark.skipif(not HEAVY_CLIP.exists(), reason="needs the dataset")
def test_settings_find_more_than_the_originals_did():
    """The shipped settings must beat the original 960/0.25 pair on a heavy frame.

    960 at 0.25 was not a measured optimum: 960 was the top of the sweep's range, and 0.25
    was the tracker's high threshold used as a detector threshold. Together they returned
    19 vehicles on this frame against 26 for the settings that replaced them, at identical
    cost per frame. Compared as a pair, because either lever alone can buy the recall.
    """
    from trafficflow.detector import load_model
    from trafficflow.video import read_frame

    config = DetectorConfig()
    frame = read_frame(HEAVY_CLIP)
    model = load_model(config)

    def found(imgsz: float, conf: float) -> int:
        result = model.predict(frame, imgsz=imgsz, conf=conf,
                               classes=config.class_ids, verbose=False)[0]
        return len(result.boxes)

    here = found(config.imgsz, config.confidence)
    originally = found(960, 0.25)
    assert originally, "vehicles should be found even with the original settings"
    assert here >= originally * 1.25, (
        f"imgsz={config.imgsz} conf={config.confidence} finds {here} vehicles against "
        f"{originally} for the original 960/0.25: not worth changing"
    )
