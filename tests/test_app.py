"""The website's API, end to end with a real clip."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ultralytics")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402
from trafficflow.config import Paths  # noqa: E402
from trafficflow.detector import DetectorConfig  # noqa: E402
from trafficflow.tracks import write_settings, write_tracks  # noqa: E402

CLIP = "cctv052x2004080516x01640"
HAS_DATA = (Paths().archive / "video" / f"{CLIP}.avi").exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="needs the dataset")


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """The real dataset and models, but a throwaway output tree.

    Uploads are kept now rather than pruned, so a test that posts a video would leave a
    folder in the real ``output/uploads`` and it would show up in the running app. Only
    ``output`` is redirected: ``archive``, the calibration and the model still come from
    the project, because these tests analyse real clips.
    """
    out = tmp_path_factory.mktemp("client-output")

    class ScratchPaths(Paths):
        @property
        def output(self):
            return out

    where = ScratchPaths()
    where.tracks.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(where))


def read_events(client: TestClient, session_id: str) -> list[tuple[str, dict]]:
    events, kind = [], None
    with client.stream("GET", f"/api/live/{session_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                events.append((kind, json.loads(line[6:])))
                if kind in ("end", "error"):
                    break
    return events


def test_the_page_is_served(client):
    assert "Traffic Flow" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


@needs_data
def test_the_clip_list_has_every_clip(client):
    clips = client.get("/api/clips").json()
    assert len(clips) == 254
    assert set(clips[0]) == {"name", "file", "traffic_class", "cached"}


def test_neither_clip_nor_file_is_refused(client):
    assert client.post("/api/live").status_code == 400


def test_unknown_clip_and_session_are_404(client):
    assert client.post("/api/live", data={"clip": "nope"}).status_code == 404
    assert client.get("/api/live/nope/events").status_code == 404


@needs_data
def test_a_clip_is_analysed_live_and_frames_can_be_replayed(client):
    started = client.post("/api/live", data={"clip": CLIP}).json()
    assert started["metric"] is True

    events = read_events(client, started["id"])
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "end", events[-1]
    frames = [data["n"] for kind, data in events if kind == "frame"]
    stats = [data for kind, data in events if kind == "stats"]
    assert len(frames) >= 40
    assert stats and stats[-1]["vehicles_counted"] is not None
    assert all(s["congestion"] for s in stats), "the video is analysed first, so every second has a class"
    assert kinds.index("end") > max(i for i, k in enumerate(kinds) if k == "frame"), "frames come before end"

    jpeg = client.get(f"/api/live/{started['id']}/frames/{frames[-1]}.jpg")
    assert jpeg.status_code == 200 and jpeg.headers["content-type"] == "image/jpeg"


@pytest.mark.slow
@needs_data
def test_an_uploaded_video_is_accepted(client):
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()
    started = client.post("/api/live", files={"file": ("any_camera.avi", data)})
    assert started.status_code == 200
    assert read_events(client, started.json()["id"])[-1][0] == "end"


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """An app whose whole output tree is temporary, so the real output/ is never touched.

    This redirects ``output`` rather than ``tracks`` alone. Overriding one folder left
    every other one -- videos, figures, reports, parameters -- pointing at the real
    project, and a test that cleared the cache deleted the batch run's artefacts for
    real. Sandbox the root of the tree, not a leaf of it.
    """
    out = tmp_path_factory.mktemp("output")

    class ScratchPaths(Paths):
        @property
        def output(self):
            return out

    where = ScratchPaths()
    where.tracks.mkdir(parents=True, exist_ok=True)
    return TestClient(create_app(where)), where.tracks


def cached(client: TestClient, clip: str) -> bool:
    return next(c["cached"] for c in client.get("/api/clips").json() if c["name"] == clip)


@needs_data
def test_removing_one_clip_deletes_only_its_saved_tracks(scratch):
    client, tracks = scratch
    other = "cctv052x2004080516x01638"
    for name in (CLIP, other):
        (tracks / f"{name}.csv").write_text("frame,track_id,cls,conf,x1,y1,x2,y2\n", encoding="utf-8")
        # A table only counts as cached when it records the current detector settings.
        write_settings(tracks / f"{name}.csv", DetectorConfig())
    assert cached(client, CLIP)

    assert client.delete(f"/api/cache/{CLIP}").json() == {"removed": 1}
    assert not (tracks / f"{CLIP}.csv").exists() and (tracks / f"{other}.csv").exists()
    assert not cached(client, CLIP)
    assert (Paths().archive / "video" / f"{CLIP}.avi").exists(), "the video is kept"
    assert client.delete(f"/api/cache/{CLIP}").json() == {"removed": 0}
    assert client.delete("/api/cache/nope").status_code == 404


@needs_data
def test_clearing_removes_every_saved_track(scratch):
    client, tracks = scratch
    for name in ("a", "b", "c"):
        (tracks / f"{name}.csv").write_text("frame,track_id,cls,conf,x1,y1,x2,y2\n", encoding="utf-8")
    assert client.delete("/api/cache").json()["removed"] >= 3
    assert not list(tracks.glob("*.csv"))
    assert (Paths().archive / "video" / f"{CLIP}.avi").exists()


@pytest.mark.slow
@needs_data
def test_a_clip_without_saved_tracks_is_detected_and_saved_for_next_time(scratch):
    client, tracks = scratch
    (tracks / f"{CLIP}.csv").unlink(missing_ok=True)
    started = client.post("/api/live", data={"clip": CLIP}).json()
    assert started["cached_tracks"] is False
    assert read_events(client, started["id"])[-1][0] == "end"
    assert (tracks / f"{CLIP}.csv").exists() and cached(client, CLIP)
    assert client.post("/api/live", data={"clip": CLIP}).json()["cached_tracks"] is True


def test_results_explain_how_to_produce_them_when_missing(client):
    """Results follow the saved tables, not the batch run's parameter file.

    Asks the app under test what it holds rather than looking at the real output folder:
    the fixture is sandboxed, so the two disagree.
    """
    client.delete("/api/cache")
    response = client.get("/api/results")
    assert response.status_code == 404, "nothing saved, so there is nothing to measure"
    assert "run.py detect" in response.json()["detail"]


def write_synthetic_tracks(path, vehicles: int = 4, frames: int = 40) -> None:
    """A few cars down the frame: enough to be measured, not real enough to be fast.

    These boxes are placed in pixels rather than projected back through the site
    homography, so the speeds they imply are not reliable and the scatter stays empty.
    The clip and level-of-service counts still follow them, which is what this file
    asserts on; the numbers themselves are the pipeline tests' job.
    """
    rows = ["frame,track_id,cls,conf,x1,y1,x2,y2"]
    for vehicle in range(1, vehicles + 1):
        for frame in range(frames):
            x, y = 110 + vehicle * 28, 40 + frame * 4 + vehicle * 9
            rows.append(f"{frame},{vehicle},2,0.9,{x},{y},{x + 24},{y + 18}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@needs_data
def test_results_follow_the_saved_track_tables(scratch):
    """Results are measured from the tables on disk: delete one and it leaves the page."""
    client, tracks = scratch
    first, second = "cctv052x2004080516x01638", "cctv052x2004080516x01640"

    client.delete("/api/cache")   # the fixture is shared, so start from a known-empty one
    assert client.get("/api/results").status_code == 404, "no tables, no results"

    for name in (first, second):
        write_synthetic_tracks(tracks / f"{name}.csv")
    body = client.get("/api/results").json()
    assert body["clips"] == 2
    assert sum(body["los"].values()) == 2

    assert client.delete(f"/api/cache/{first}").json() == {"removed": 1}
    body = client.get("/api/results").json()
    assert body["clips"] == 1, "a deleted table leaves the results"
    assert sum(body["los"].values()) == 1

    write_synthetic_tracks(tracks / f"{first}.csv")
    assert client.get("/api/results").json()["clips"] == 2, "a new table joins the results"

    client.delete("/api/cache")
    assert client.get("/api/results").status_code == 404


# -- replacing and stopping a session ---------------------------------------------


@needs_data
def test_starting_a_second_video_retires_the_first(client):
    """Only one session at a time, and the old one is gone before the new one runs.

    Detection is CPU-bound and every session shares the one loaded YOLO model, so two
    live workers must never overlap. The old session's id stops resolving, which is
    how the page learns to stop asking about it.
    """
    first = client.post("/api/live", data={"clip": CLIP}).json()
    second = client.post("/api/live", data={"clip": CLIP}).json()

    assert first["id"] != second["id"]
    assert client.get(f"/api/live/{first['id']}/events").status_code == 404
    assert read_events(client, second["id"])[-1][0] == "end"


@needs_data
def test_stopping_a_session_ends_it(client):
    started = client.post("/api/live", data={"clip": CLIP}).json()
    stopped = client.post(f"/api/live/{started['id']}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["state"] in ("stopped", "done", "running", "starting")

    # Whatever it was doing, the stream terminates rather than hanging.
    assert read_events(client, started["id"])[-1][0] in ("end", "error")


@needs_data
def test_a_dataset_clip_is_flagged_as_in_sample(client):
    """The CNN was fine-tuned on all 254 clips, so a live answer on one is not evidence."""
    started = client.post("/api/live", data={"clip": CLIP}).json()
    assert started["in_sample"] is True
    assert started["uploaded"] is False
    client.post(f"/api/live/{started['id']}/stop")


@needs_data
def test_an_upload_is_not_flagged_as_in_sample(client):
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()
    started = client.post("/api/live", files={"file": ("elsewhere.avi", data)}).json()
    assert started["in_sample"] is False
    assert started["uploaded"] is True
    client.post(f"/api/live/{started['id']}/stop")


@needs_data
def test_a_bad_upload_does_not_disturb_a_running_session(client):
    """Validation happens before the running session is touched.

    The upload used to be written and the previous session stopped *before* anything
    checked the file was a video, so a mistyped drag-and-drop killed a good session
    and then returned 400.

    A session has to actually be running for that to mean anything, so one is started
    first. ``stop`` is the probe: it resolves the id through the same lookup the rest
    of the API uses, and answers 404 once a session has been retired.
    """
    running = client.post("/api/live", data={"clip": CLIP}).json()

    refused = client.post("/api/live", files={"file": ("notes.txt", b"not a video")})
    assert refused.status_code == 400
    assert "video" in refused.json()["detail"]

    still_there = client.post(f"/api/live/{running['id']}/stop")
    assert still_there.status_code == 200, "the bad upload retired the running session"


@needs_data
def test_clearing_never_touches_the_batch_artefacts_or_the_dataset(scratch):
    """Clear all empties the app's videos, and nothing else.

    It once emptied every folder under output/, which destroyed the annotated videos,
    the figures, the report and parameters/clips.csv -- artefacts that cost a ~1.5 hour
    detection run to rebuild. This pins the blast radius to tracks and uploads.
    """
    client, tracks = scratch
    out = tracks.parent

    keepers = {}
    for folder, name in (("videos", "annotated.mp4"), ("figures", "diagram.png"),
                         ("reports", "insights.md"), ("parameters", "clips.csv")):
        (out / folder).mkdir(parents=True, exist_ok=True)
        target = out / folder / name
        target.write_text("precious", encoding="utf-8")
        keepers[target] = "precious"

    write_synthetic_tracks(tracks / f"{CLIP}.csv")
    (out / "uploads" / "abc123").mkdir(parents=True, exist_ok=True)
    (out / "uploads" / "abc123" / "tracks.csv").write_text("x", encoding="utf-8")

    client.delete("/api/cache")

    assert not list(tracks.glob("*.csv")), "saved detections are removed"
    assert not list((out / "uploads").iterdir()), "uploaded videos are removed"
    for target, content in keepers.items():
        assert target.exists() and target.read_text(encoding="utf-8") == content, \
            f"{target.parent.name}/{target.name} must survive Clear all"
    assert (Paths().archive / "video" / f"{CLIP}.avi").exists(), "the dataset is never touched"
    assert (Paths().models / "congestion_cnn.onnx").exists(), "the model is never touched"


@pytest.mark.slow
@needs_data
def test_an_upload_is_kept_listed_and_removable(client):
    """An uploaded video persists, reaches the results, and Remove takes it away again."""
    client.delete("/api/cache")
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()

    started = client.post("/api/live", files={"file": ("mine.avi", data)})
    assert started.status_code == 200
    assert read_events(client, started.json()["id"])[-1][0] == "end"

    listed = client.get("/api/uploads").json()
    assert len(listed) == 1 and listed[0]["name"] == "mine.avi" and listed[0]["cached"]

    body = client.get("/api/results").json()
    assert body["clips"] == 1
    assert body["points"][0]["traffic_class"] is None, "an upload has no hand-given class"
    assert body["by_hour"] == [], "an upload has no hour, so there is no hourly profile"

    assert client.delete(f"/api/uploads/{listed[0]['id']}").json() == {"removed": 1}
    assert client.get("/api/uploads").json() == []
    assert client.get("/api/results").status_code == 404, "its results go with it"


def test_an_unknown_upload_is_404(client):
    assert client.delete("/api/uploads/zzzz").status_code == 404
    assert client.post("/api/live", data={"upload": "zzzz"}).status_code == 404


# -- storing uploads without analysing them ------------------------------------------
#
# The Live page is upload-only and takes several files at once, so the browser stores
# the whole selection first and then analyses them one at a time. Storing therefore has
# to be its own step: posting five files to /api/live would start five sessions, and
# four of them would be stopped by the fifth before they had read a frame.

@needs_data
def test_storing_an_upload_does_not_start_a_session(client):
    client.delete("/api/uploads")
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()

    stored = client.post("/api/uploads", files={"file": ("queued.avi", data)})
    assert stored.status_code == 200
    body = stored.json()
    assert body["name"] == "queued.avi" and body["id"]

    listed = client.get("/api/uploads").json()
    assert [(u["id"], u["name"], u["cached"]) for u in listed] == [(body["id"], "queued.avi", False)]

    # Nothing is analysing, and the stored upload can be played later by its id.
    assert client.get(f"/api/live/{body['id']}/events").status_code == 404
    client.delete("/api/uploads")


def test_storing_something_that_is_not_a_video_is_rejected(client):
    stored = client.post("/api/uploads", files={"file": ("notes.txt", b"not a video at all")})
    assert stored.status_code == 400
    assert client.get("/api/uploads").json() == [], "the rejected file leaves nothing behind"


@needs_data
def test_clearing_uploads_keeps_the_batch_runs_track_tables(scratch):
    """The page's Clear button must not be able to delete the detection pass.

    ``DELETE /api/cache`` removes the saved tracks of all 254 dataset clips along with
    the uploads. Those cost hours to rebuild and, now that the Live page has no dataset
    picker, there is no way to rebuild them from the browser at all. So the button uses
    this endpoint, which only ever touches uploads.
    """
    client, tracks = scratch
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()
    client.post("/api/uploads", files={"file": ("one.avi", data)})
    client.post("/api/uploads", files={"file": ("two.avi", data)})
    batch_table = tracks / f"{CLIP}.csv"
    batch_table.write_text("frame,track_id,cls,conf,x1,y1,x2,y2\n", encoding="utf-8")

    assert client.delete("/api/uploads").json() == {"removed": 2}
    assert client.get("/api/uploads").json() == []
    assert batch_table.exists(), "the dataset clip's detection survives"


@needs_data
def test_saved_results_survive_a_server_restart(tmp_path):
    """A restart is a new app on the same output tree: every saved result must still be there."""
    class ScratchPaths(Paths):
        @property
        def output(self):
            return tmp_path

    where = ScratchPaths()
    data = (Paths().archive / "video" / f"{CLIP}.avi").read_bytes()
    with TestClient(create_app(where)) as first:
        upload = first.post("/api/uploads", files={"file": ("kept.avi", data)}).json()["id"]
        table = where.tracks / f"{CLIP}.csv"
        write_tracks(table, [(1, 1, 2, 0.9, 10.0, 20.0, 30.0, 40.0)])
        write_settings(table, DetectorConfig())
        write_tracks(where.uploads / upload / "tracks.csv", [(1, 1, 2, 0.9, 10.0, 20.0, 30.0, 40.0)])
        saved = table.read_bytes()
    # Leaving the block runs the shutdown hook, as Ctrl+C on the server does.

    with TestClient(create_app(where)) as second:
        assert table.read_bytes() == saved
        assert cached(second, CLIP)
        assert [(u["id"], u["cached"]) for u in second.get("/api/uploads").json()] == [(upload, True)]


def test_a_failed_rewrite_keeps_the_saved_table(tmp_path, monkeypatch):
    """Stopping the server mid-save must not leave a truncated table in place of a good one."""
    import trafficflow.tracks as tracks_module

    table = tmp_path / "clip.csv"
    write_tracks(table, [(1, 1, 2, 0.9, 10.0, 20.0, 30.0, 40.0)])
    saved = table.read_bytes()

    def interrupted(*_args):
        raise KeyboardInterrupt
    monkeypatch.setattr(tracks_module.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_tracks(table, [(2, 2, 2, 0.5, 1.0, 2.0, 3.0, 4.0)])
    assert table.read_bytes() == saved
