"""One analysis of a video: analyse everything first, then hand the page a finished video.

Pass 1 -- analyse. Every frame is tracked (ByteTrack needs every frame). For a dataset
clip the tracks in ``output/tracks/`` are reused, so this takes seconds; a dataset clip
without them runs YOLO here and writes them there for next time, and an uploaded video
runs YOLO every time and writes nothing. Every second of video the traffic is measured
on the tracks so far and the congestion CNN classifies three frames a second apart.

Pass 2 -- draw. With the whole video measured, every frame is drawn with each vehicle's
box, type and final speed. Nothing is drawn on the road itself. The page shows a progress
bar during both passes and then plays the finished frames.

Videos are at most MAX_SECONDS long, so everything stays in memory. Any uploaded video is
resized to 320x240 and measured with the I-5 calibration -- a decision, not a detection
of the camera. On another camera the counts, types and tracking still hold, but speed,
density and level of service do not, so ``info()`` marks uploads and the page says so.
"""

from __future__ import annotations

import json
import threading
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from trafficflow import insights
from trafficflow.config import Calibration, SiteConfig
from trafficflow.congestion import CongestionModel, load_congestion_model
from trafficflow.detector import DetectorConfig, FrameTracker, load_model
from trafficflow.overlay import annotate_frame
from trafficflow.pipeline import measure, road_frame_from
from trafficflow.tracks import VEHICLE_CLASSES, TrackTable, stale_fields, write_settings, write_tracks
from trafficflow.video import iter_frames

__all__ = ["Models", "LiveSession", "MAX_SECONDS", "probe_video"]

#: Longest stretch of video analysed. Uploads are expected to be 5-6 s.
MAX_SECONDS = 10.0

#: Frames are drawn three times larger so labels are readable.
SCALE = 3

#: How long :meth:`LiveSession.stop` waits for the worker to notice. Long enough for a
#: single YOLO frame to finish on CPU, short enough not to hold up the web request that
#: is starting the next video.
JOIN_TIMEOUT = 5.0


def probe_video(video: Path, fallback_fps: float) -> tuple[float, int]:
    """Frame rate and how many frames will actually be analysed.

    Split out of :class:`LiveSession` because an upload is now stored before anything
    analyses it, and the metadata written at that moment has to agree with what the
    session later reports. Two copies of the ``MAX_SECONDS`` arithmetic would have
    drifted the first time one of them changed.

    Raises
    ------
    ValueError
        If the file cannot be decoded as a video.
    """
    capture = cv2.VideoCapture(str(video))
    try:
        readable = capture.isOpened() and capture.read()[0]
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not readable:
        raise ValueError("this file could not be read as a video")

    fps = fps if fps and fps > 0 else fallback_fps
    limit = int(MAX_SECONDS * fps)
    return fps, (min(count - 1, limit) if count > 1 else limit)   # frame 1 is skipped


@dataclass
class Models:
    """The detector and the congestion CNN, loaded once so the first click does not wait."""

    detector: object
    congestion: CongestionModel | None

    @classmethod
    def load(cls) -> "Models":
        try:
            congestion = load_congestion_model()
        except (FileNotFoundError, KeyError, ImportError):
            congestion = None   # the page shows "no model" instead of failing
        return cls(load_model(DetectorConfig()), congestion)


class LiveSession:
    """One video being analysed in a background thread."""

    def __init__(self, video: Path, name: str, site: SiteConfig, calibration: Calibration,
                 models: Models, tracks: Path | None = None,
                 uploaded: bool = False) -> None:
        self.fps, self.total_frames = probe_video(Path(video), site.image.fps)

        self.id = uuid.uuid4().hex[:12]
        self.video, self.name, self.site, self.models = Path(video), name, site, models
        self.uploaded = uploaded
        self.road = road_frame_from(site, calibration, day_index=1 if "x20040805" in name else 2)
        #: Tracks already saved for a dataset clip; None means run YOLO. A table saved by
        #: a different detector is not a cache -- replaying it would show the old
        #: detector's boxes while the page claims to be running the current one.
        usable = (tracks is not None and tracks.exists()
                  and not stale_fields(tracks, DetectorConfig()))
        self.cached = TrackTable.load(tracks) if usable else None
        #: Where a dataset clip's fresh detection is saved, so the next play reuses it.
        self.save_to = tracks if tracks is not None and self.cached is None else None
        #: An upload's metadata, where its whole-clip class is kept for the sidebar.
        self.meta = tracks.parent / "meta.json" if uploaded and tracks is not None else None

        self.state = "starting"          # starting | running | done | stopped | failed
        self.error = ""
        self.jpegs: dict[int, bytes] = {}
        self.events: list[tuple[str, dict]] = []
        self._stop = threading.Event()
        self._changed = threading.Condition()
        self._thread: threading.Thread | None = None

    # -- control ----------------------------------------------------------------

    def info(self) -> dict:
        """What the page needs to know before the first frame arrives.

        ``in_sample`` is a disclosure, not a setting. The exported congestion model was
        fine-tuned on all 254 dataset clips, so when one of them is played back here the
        CNN is being asked about a clip it was trained on and will look better than it
        is. The honest figure is the cross-validated one on the Results tab, which holds
        whole recordings out and is unaffected by this. An uploaded video is genuinely
        unseen, so the flag is false there.
        """
        return {"id": self.id, "name": self.name, "fps": self.fps, "total_frames": self.total_frames,
                "metric": True, "cached_tracks": self.cached is not None,
                "uploaded": self.uploaded, "in_sample": not self.uploaded}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = JOIN_TIMEOUT) -> None:
        """Ask the worker to stop, and wait for it to actually let go.

        Setting the event is not enough on its own: the caller is usually about to
        start another session on the *same* YOLO model, and two threads stepping
        through one tracker is how track ids get mixed between videos. So this joins.

        The join is bounded because it runs inside a web request. A worker that has
        not noticed within ``timeout`` is left to finish on its own -- it is a daemon
        thread, its session has already been dropped from the registry, and nothing
        reads its output any more.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def events_after(self, index: int, timeout: float = 15.0) -> list[tuple[str, dict]]:
        """Events from position ``index`` on, waiting up to ``timeout`` for the first."""
        with self._changed:
            self._changed.wait_for(lambda: len(self.events) > index, timeout)
            return self.events[index:]

    def _publish(self, kind: str, data: dict) -> None:
        with self._changed:
            self.events.append((kind, data))
            self._changed.notify_all()

    # -- work -------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self.state = "running"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                images, rows = self._analyse()
                if self._stop.is_set():
                    raise _Stopped   # a stopped detection is partial, so it is never saved
                if self.save_to is not None:
                    write_tracks(self.save_to, rows)
                    write_settings(self.save_to, DetectorConfig())
                seconds = self._measure_each_second(images, rows)
                self._draw(images, rows)
                self._remember_class(seconds)
            self.state = "done"
            self._publish("end", {"state": self.state, "summary": seconds[-1][1] if seconds else {}})
        except _Stopped:
            self.state = "stopped"
            self._publish("end", {"state": self.state, "summary": {}})
        except Exception as error:  # shown on the page rather than lost in a thread
            self.state, self.error = "failed", f"{type(error).__name__}: {error}"
            self._publish("error", {"message": self.error})

    def _remember_class(self, seconds: list[tuple[int, dict]]) -> None:
        """Save an upload's final CNN class into its metadata, so the sidebar can show it."""
        congestion = seconds[-1][1].get("congestion") if seconds else None
        if self.meta is None or not congestion or not self.meta.exists():
            return
        meta = json.loads(self.meta.read_text(encoding="utf-8"))
        meta["traffic_class"] = congestion["label"]
        self.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _progress(self, step: str, done: int, total: int) -> None:
        self._publish("progress", {"step": step, "done": done, "total": total})

    def _analyse(self) -> tuple[dict[int, np.ndarray], list[tuple]]:
        """Pass 1: every frame, resized to 320x240, and every track row."""
        size = (self.site.image.width, self.site.image.height)
        tracker = None if self.cached is not None else \
            FrameTracker(DetectorConfig(), self.models.detector)
        images: dict[int, np.ndarray] = {}
        rows: list[tuple] = []
        for number, image in iter_frames(self.video, max_frames=self.total_frames):
            if self._stop.is_set():
                break
            if (image.shape[1], image.shape[0]) != size:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            images[number] = image
            if tracker is not None:
                rows.extend(tracker.track(image, number, self.site.image))
                self._progress("Detecting and tracking vehicles", len(images), self.total_frames)
        if self.cached is not None:
            last = max(images, default=0)
            rows = [tuple(row) for row in self.cached.rows if row[0] <= last]
        return images, rows

    def _measure_each_second(self, images: dict[int, np.ndarray], rows: list[tuple]) -> list[tuple[int, dict]]:
        """Measure the tracks up to the end of each second, and classify that moment."""
        model = self.models.congestion
        cnn_frames = {n: model.prepare_frame(image) for n, image in images.items()} if model else {}
        gap = max(1, round(self.fps * model.spec.frame_gap / model.image.fps)) if model else 1
        every = max(1, round(self.fps))
        numbers = sorted(images)
        marks = numbers[every - 1::every] or numbers[-1:]
        if numbers and marks[-1] != numbers[-1]:
            marks.append(numbers[-1])

        table = TrackTable(np.asarray(rows, dtype=float).reshape(-1, 8))
        seconds = []
        for done, number in enumerate(marks, start=1):
            if self._stop.is_set():
                raise _Stopped
            so_far = TrackTable(table.rows[table.frames <= number])
            processed = numbers.index(number) + 1
            prediction = model.predict_frames(cnn_frames, number, gap) if model else None
            stats = self._stats(so_far, number, processed, prediction)
            seconds.append((number, stats))
            self._progress("Measuring speed, density and congestion", done, len(marks))

        # The CNN needs three frames a second apart, so it has no answer for the first two
        # seconds. The whole video is analysed before it plays, so show its first answer there.
        first = next((stats["congestion"] for _, stats in seconds if stats["congestion"]), None)
        for _, stats in seconds:
            if stats["congestion"] is None and first is not None:
                stats["congestion"] = first
                stats["insight"] = insights.live_line(stats.get("speed_median_kmh"), stats.get("level_of_service"),
                                                      first["label"], self.site.roadway.speed_limit_kmh)
            self._publish("stats", stats)
        return seconds

    def _draw(self, images: dict[int, np.ndarray], rows: list[tuple]) -> None:
        """Pass 2: draw each frame with every vehicle's box, type and final speed."""
        table = TrackTable(np.asarray(rows, dtype=float).reshape(-1, 8))
        final = measure(table, self.road, self.fps, self.site.level_of_service, n_frames=len(images)) \
            if len(table) else None
        speeds = {estimate.track_id: estimate for estimate in final.speeds} if final else {}
        for done, number in enumerate(sorted(images), start=1):
            if self._stop.is_set():
                raise _Stopped
            canvas = annotate_frame(images[number], table.frame_rows(number), speeds, scale=SCALE)
            self.jpegs[number] = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            self._publish("frame", {"n": number})
            if done % 5 == 0 or done == len(images):
                self._progress("Drawing the video", done, len(images))

    def _stats(self, table: TrackTable, number: int, processed: int, prediction) -> dict:
        seconds = round(processed / self.fps, 1)
        congestion = None if prediction is None else \
            {"label": prediction.label, "confidence": round(prediction.confidence, 3)}
        if len(table) == 0:
            return {"n": number, "t": seconds, "vehicles_counted": 0, "tracks_seen": 0, "in_view": 0,
                    "congestion": congestion, "insight": insights.live_line(None, None, None, 1.0)}

        measured = measure(table, self.road, self.fps, self.site.level_of_service, n_frames=processed)
        by_id = {estimate.track_id: estimate for estimate in measured.speeds}
        median = _number(measured.speed_summary["median_kmh"])
        los = measured.density.level_of_service
        return {
            "n": number,
            "t": seconds,
            "speed_median_kmh": median,
            "speed_p85_kmh": _number(measured.speed_summary["p85_kmh"]),
            "density_pc_mi_ln": _number(measured.density.density_pc_per_mi_per_ln),
            "level_of_service": los,
            "vehicles_counted": measured.counts.total,
            "tracks_seen": measured.counts.tracks_seen,
            "in_view": int(len(table.frame_rows(number))),
            "by_class": measured.counts.by_class,
            "lane_occupancy": {str(k): round(v, 3) for k, v in measured.occupancy.occupancy.items()},
            "lane_changes": len(measured.lane_changes),
            "congestion": congestion,
            "insight": insights.live_line(median, los, prediction.label if prediction else None,
                                          self.site.roadway.speed_limit_kmh),
            "vehicles": [_vehicle(row, by_id) for row in table.frame_rows(number)],
        }


class _Stopped(Exception):
    """Raised inside the worker when the user presses Stop."""


def _vehicle(row: np.ndarray, speeds: dict) -> dict:
    """One row of the vehicles-in-view table; speed only when its fit is reliable."""
    track_id = int(row[1])
    estimate = speeds.get(track_id)
    return {
        "id": track_id,
        "class": VEHICLE_CLASSES.get(int(row[2]), "unknown"),
        "lane": estimate.lane if estimate else None,
        "speed_kmh": _number(estimate.speed_kmh) if estimate and estimate.is_reliable else None,
    }


def _number(value: float) -> float | None:
    return None if value != value else round(float(value), 1)   # NaN != NaN
