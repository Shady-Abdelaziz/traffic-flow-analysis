"""Stopping a live session, in both passes.

Stop used to be checked only in the detection loop. The measure pass and the draw pass
ran to completion regardless, and nothing joined the worker, so pressing Stop and
immediately playing another video left two threads sharing the one loaded YOLO model.

These tests need no dataset and no models: the session is given a cached track table, so
the detector is never called, and no congestion model, so the CNN is skipped.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from app.live import LiveSession, Models, _Stopped
from trafficflow.config import Paths, load_calibration, load_site
from trafficflow.tracks import COLUMNS, write_tracks

FPS = 10.0
N_FRAMES = 12


@pytest.fixture(scope="module")
def config():
    where = Paths()
    return load_site(where.config / "site.yaml"), load_calibration(where.calibration)


@pytest.fixture
def video(tmp_path):
    """A short synthetic 320x240 clip, so nothing here depends on the archive."""
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def tracks(tmp_path):
    """A cached track table covering the clip, so the detector is never loaded."""
    path = tmp_path / "synthetic.csv"
    rows = [
        (frame, 1, 2, 0.9, 150.0, 180.0 - 2.0 * frame, 170.0, 200.0 - 2.0 * frame)
        for frame in range(2, N_FRAMES + 1)
    ]
    write_tracks(path, rows)
    assert path.exists() and len(COLUMNS) == 8
    return path


@pytest.fixture
def session(video, tracks, config):
    site, calibration = config
    return LiveSession(video, "synthetic", site, calibration,
                       Models(detector=None, congestion=None), tracks=tracks)


def analysed(session):
    """Run pass 1 only, so the later passes can be driven directly."""
    return session._analyse()


# -- each pass notices ------------------------------------------------------------


def test_the_measure_pass_stops(session):
    images, rows = analysed(session)
    session.stop()
    with pytest.raises(_Stopped):
        session._measure_each_second(images, rows)


def test_the_draw_pass_stops(session):
    """Not just eventually: the loop gives up where it is, leaving the rest undrawn."""
    images, rows = analysed(session)
    session.stop()
    with pytest.raises(_Stopped):
        session._draw(images, rows)
    assert len(session.jpegs) < len(images)


def test_an_unstopped_session_completes_both_passes(session):
    """The control: without Stop, the same passes run to the end."""
    images, rows = analysed(session)
    session._measure_each_second(images, rows)
    session._draw(images, rows)
    assert len(session.jpegs) == len(images)


# -- the session as a whole -------------------------------------------------------


def test_a_session_stopped_before_it_starts_reports_stopped(session):
    session.stop()
    session._run()
    assert session.state == "stopped"
    assert session.events[-1] == ("end", {"state": "stopped", "summary": {}})


def test_stop_waits_for_the_worker_to_let_go(session):
    """The reason Stop joins: the next session reuses the one shared YOLO model."""
    session.start()
    session.stop()
    assert session._thread is not None
    assert not session._thread.is_alive()


def test_stop_is_safe_before_start_and_twice(session):
    session.stop()
    session.stop()
    assert session._thread is None


def test_stop_from_inside_the_worker_does_not_deadlock(session):
    """A join on the current thread would hang forever, so it is skipped."""
    session._thread = threading.current_thread()
    session.stop(timeout=0.1)
    assert session._stop.is_set()


def test_a_completed_session_reports_done(session):
    session.start()
    assert session._thread is not None
    session._thread.join(30)
    assert session.state == "done"
    assert len(session.jpegs) > 0
