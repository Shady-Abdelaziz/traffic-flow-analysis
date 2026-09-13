"""Detection and tracking: the one expensive stage, run once over all 254 clips.

Uses YOLO with COCO weights, restricted to COCO's road-vehicle classes, and ByteTrack for
identity across frames. The detector is **not** fine-tuned, and could not be: the
database ships no bounding boxes at all. COCO already separates car, motorcycle, bus and
truck, which is exactly the classification the exam asks for.

**Checkpoint.** ``yolo11n``, for the batch pass and the live page: a pass over all 254 clips
takes about 25 minutes on a laptop CPU and a live upload takes seconds. ``yolo26m`` finds
more of the cars ``yolo11n`` misses, but costs roughly ten times the compute per frame, so
it is used only for the three demo clips (``python run.py demo``). Every track table
records the checkpoint that made it in a ``.json`` beside it. See :class:`DetectorConfig`.

**Input size.** The clips are 320x240 and distant vehicles are only a few pixels across,
so frames are letterboxed up before inference. 640 is used because it keeps a live upload
to seconds rather than minutes on a CPU; its cost is the most distant vehicles.
:func:`compare_input_sizes` measures what each size keeps against another on real frames
(``python run.py detect --compare-sizes``).

**The roadway mask.** The opposing northbound carriageway is visible in the upper left of
every frame. Those vehicles are real and the detector finds them correctly, but they
belong to the other direction and would inflate southbound counts and density. They are
dropped by the same ``x >= roadway_crop_x`` boundary the classifier crops to -- and, with
``crop_to_roadway``, never sent to the detector at all.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trafficflow.config import ImageGeometry, Paths
from trafficflow.dataset import Clip, TrafficDatabase
from trafficflow.tracks import VEHICLE_CLASSES, stale_fields, write_settings, write_tracks
from trafficflow.video import iter_frames

__all__ = [
    "DetectorConfig", "FrameTracker", "load_model", "on_roadway",
    "compare_input_sizes", "detect_clip", "detect_all",
]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Detector and tracker settings.

    Attributes
    ----------
    weights:
        COCO-pretrained checkpoint. ``yolo11n``, for speed: about 6.5 GFLOPs at 640 against
        68 for ``yolo26m``, so the pass over all 254 clips takes about 25 minutes on a laptop
        CPU and a live upload takes seconds. ``yolo26m`` finds more of the clear cars
        ``yolo11n`` misses and reads fewer cars as trucks; ``run.py demo`` uses it for the
        three demo clips. A table made with other weights counts as stale: ``run.py detect``
        and the live page re-detect it rather than reuse it.
    imgsz:
        Inference size; frames are letterboxed up to it. 640, so a live upload takes
        seconds on a CPU rather than minutes. The cost is the most distant vehicles, the
        first to fall below what the model can resolve. How much 640 loses against 960 has
        not been measured on this footage; ``python run.py detect --compare-sizes`` does
        it. Set here and nowhere else, so the live page and the batch pass cannot drift
        apart.
    confidence:
        Detection threshold, and it must stay at or below the tracker's
        ``track_low_thresh``. This is a contract with ByteTrack, not a taste setting.

        ByteTrack associates in two stages: strong boxes (>= ``track_high_thresh``) match
        first, then the weak band between ``track_low_thresh`` and ``track_high_thresh``
        is used to recover tracks that would otherwise be lost. Passing 0.25 here -- the
        tracker's own high threshold -- deleted that weak band before the tracker ever saw
        it, so the second stage ran on nothing. A stationary, plainly visible car whose
        score wobbled across 0.25 between frames blinked out and back, and the median
        track lasted 8 frames.

        0.10 matches ``track_low_thresh`` in ``bytetrack.yaml``. The weak band cannot
        invent vehicles: ``new_track_thresh`` is 0.25, so a low-scoring box may only
        continue a track that already exists, never start one.
    iou:
        NMS overlap threshold: a box is suppressed when it overlaps a higher-scoring box
        of the same class by more than this. Higher means *less* suppression.

        0.5, which is below ultralytics' own default of 0.7. The reasoning for raising it
        was that vehicles seen from behind on a curve overlap in the image, so a real car
        behind its neighbour could be dropped as a duplicate. 0.7 was tried on this footage
        and looked worse on screen, so it was reverted. Lowering ``confidence`` to 0.10 is
        what actually reduced the blinking; this knob did not help and 0.5 stands until
        someone has a measurement rather than an argument.
    tracker:
        Ultralytics tracker config. ByteTrack keeps low-confidence boxes as candidates
        for association, which suits small, faint, distant vehicles.
    crop_to_roadway:
        Cut each frame at ``roadway_crop_x`` *before* YOLO sees it. Everything left of
        that line is the northbound carriageway, whose boxes :func:`on_roadway` throws
        away anyway, so detecting there was wasted work: the crop is 208 of 320 columns,
        about 35% fewer pixels per frame.

        ``imgsz`` keeps its meaning -- the size the *full* frame would be inferred at.
        YOLO scales an image's longest side to ``imgsz``, so passing 640 with the narrower
        crop would scale it 2.67x instead of 2x and cost *more* than the full frame. The
        tracker instead infers the crop at the full frame's scale (480 for 640), so every
        vehicle is exactly as many pixels across as before. See :func:`crop_imgsz`.

        Recorded in each table's settings, so a table made without the crop is re-detected.
    """

    weights: str = "yolo11n.pt"
    imgsz: int = 640
    confidence: float = 0.10
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    crop_to_roadway: bool = True

    @property
    def class_ids(self) -> list[int]:
        return sorted(VEHICLE_CLASSES)


def load_model(config: DetectorConfig):
    """Import ultralytics lazily; use <weights> from the project root when present, else let ultralytics fetch it."""
    from ultralytics import YOLO

    local = Paths().models / config.weights
    return YOLO(str(local) if local.exists() else config.weights)


def on_roadway(boxes: np.ndarray, image: ImageGeometry) -> np.ndarray:
    """Mask of detections whose ground point lies on the southbound roadway.

    Tested at the box's bottom-centre -- the tyre contact patch -- so a vehicle is
    judged by where it touches the road, not by where its roof happens to overhang.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    return centre_x >= image.roadway_crop_x


def crop_imgsz(imgsz: int, image: ImageGeometry, stride: int = 32) -> int:
    """Inference size for the roadway crop that keeps the full frame's scale.

    The full frame is scaled by ``imgsz / max(width, height)``; the crop's longest side
    gets that same factor, rounded up to the model stride. 640 on 320x240 gives 480.
    """
    scale = imgsz / max(image.width, image.height)
    longest = max(image.width - image.roadway_crop_x, image.height)
    return -(-round(scale * longest) // stride) * stride


class FrameTracker:
    """YOLO + ByteTrack fed one frame at a time.

    ByteTrack matches boxes between consecutive frames, so every frame must go through
    it -- sampling frames breaks the identities speed is measured from. The batch pass
    and the live page both use this class, so there is one tracker in the project.
    """

    def __init__(self, config: DetectorConfig | None = None, model=None) -> None:
        self.config = config or DetectorConfig()
        self.model = model if model is not None else load_model(self.config)
        self.model.predictor = None  # a fresh tracker: ids never carry over between videos

    def track(self, image: np.ndarray, number: int, image_geometry: ImageGeometry) -> list[tuple]:
        """Track rows for one frame, southbound roadway only.

        ``image`` is the whole frame and the rows are in its coordinates; with
        ``crop_to_roadway`` only the part right of ``roadway_crop_x`` goes to YOLO.
        """
        settings = self.config
        left, size = 0, settings.imgsz
        if settings.crop_to_roadway:
            left, size = image_geometry.roadway_crop_x, crop_imgsz(settings.imgsz, image_geometry)
            image = np.ascontiguousarray(image[:, left:])
        result = self.model.track(
            image, imgsz=size, conf=settings.confidence, iou=settings.iou,
            classes=settings.class_ids, tracker=settings.tracker, persist=True, verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        xyxy[:, [0, 2]] += left  # back to full-frame x
        keep = on_roadway(xyxy, image_geometry)
        ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        return [
            (number, int(i), int(c), float(p), *box.tolist())
            for box, i, c, p in zip(xyxy[keep], ids[keep], classes[keep], confidences[keep])
        ]


def compare_input_sizes(
    clips: Sequence[Clip],
    data_root: Path | str,
    image: ImageGeometry,
    sizes: Sequence[int] = (640, 960, 1280, 1536),
    config: DetectorConfig | None = None,
    weights: Sequence[str] | None = None,
    frames_per_clip: int = 3,
) -> dict[tuple[str, int], dict[str, float]]:
    """Measure what each model and input size finds, and what it costs.

    Run this before committing to a full pass. The clips are 320x240, so it is tempting
    to infer at 320 -- but vehicles at the far end of the measurement zone are 4-6 px
    across and simply vanish at that size. Upscaling recovers them.

    Keep the largest size in ``sizes`` above :attr:`DetectorConfig.imgsz`. When the two
    are equal the sweep can only ever confirm the setting it was asked to test.

    A count alone cannot justify a cheaper setting: at ``confidence`` 0.10 a weaker model
    can return as many boxes while missing the distant vehicles and adding junk. So every
    candidate is also scored against the configured setting (``config``) on the same
    frames -- ``kept`` is the share of its roadway boxes the candidate still finds -- and
    by the smallest box it returns, which is how far down the road it can see.

    Parameters
    ----------
    clips:
        A handful of clips, ideally spanning light and heavy traffic.
    sizes:
        Input sizes to compare.
    weights:
        Checkpoints to compare. Defaults to the configured one only.
    frames_per_clip:
        Frames spread evenly through each clip. One frame per clip under-samples: near
        traffic drives out of shot during a clip and the far zone fills up.

    Returns
    -------
    dict
        Keyed by ``(weights, size)``, each holding ``on_roadway``, ``cars``, ``bus_truck``,
        ``kept``, ``smallest_px`` and ``seconds_per_frame``.
    """
    import time

    settings = config or DetectorConfig()
    frames = [frame for clip in clips for frame in _spread_frames(clip, data_root, frames_per_clip)]

    def roadway_boxes(model, frame: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = model.predict(frame, imgsz=size, conf=settings.confidence, iou=settings.iou,
                                   classes=settings.class_ids, verbose=False)[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        classes = prediction.boxes.cls.cpu().numpy().astype(int)
        keep = on_roadway(boxes, image)
        return boxes[keep], classes[keep]

    reference_model = load_model(settings)
    reference = [roadway_boxes(reference_model, frame, settings.imgsz)[0] for frame in frames]

    results: dict[tuple[str, int], dict[str, float]] = {}
    for name in weights or (settings.weights,):
        model = load_model(replace(settings, weights=name))
        model.predict(frames[0], imgsz=settings.imgsz, verbose=False)  # warm up
        for size in sizes:
            found, started = [], time.perf_counter()
            for frame in frames:
                found.append(roadway_boxes(model, frame, size))
            elapsed = time.perf_counter() - started

            boxes = [b for b, _ in found]
            classes = np.concatenate([c for _, c in found])
            heights = np.concatenate([b[:, 3] - b[:, 1] for b in boxes])
            matched = sum(_matched(ref, b) for ref, b in zip(reference, boxes))
            results[(name, size)] = {
                "on_roadway": len(classes),
                "cars": int((classes == 2).sum()),
                "bus_truck": int(np.isin(classes, (5, 7)).sum()),
                "kept": matched / max(1, sum(len(ref) for ref in reference)),
                "smallest_px": float(heights.min()) if len(heights) else float("nan"),
                "seconds_per_frame": elapsed / len(frames),
            }
    return results


def _matched(reference: np.ndarray, candidate: np.ndarray, threshold: float = 0.5) -> int:
    """How many reference boxes some candidate box overlaps at IoU >= ``threshold``."""
    if len(reference) == 0 or len(candidate) == 0:
        return 0
    top_left = np.maximum(reference[:, None, :2], candidate[None, :, :2])
    bottom_right = np.minimum(reference[:, None, 2:], candidate[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    union = area(reference)[:, None] + area(candidate)[None, :] - overlap
    return int(((overlap / union) >= threshold).any(axis=1).sum())


def _spread_frames(clip: Clip, data_root: Path | str, count: int) -> list[np.ndarray]:
    from trafficflow.video import FIRST_USABLE_FRAME, probe

    path = clip.video_path(data_root)
    usable = probe(path).frame_count - (FIRST_USABLE_FRAME - 1)
    return [image for _, image in iter_frames(path, max_frames=count, step=max(1, usable // count))]


def detect_clip(
    clip: Clip,
    data_root: Path | str,
    image: ImageGeometry,
    config: DetectorConfig | None = None,
    model=None,
) -> list[tuple]:
    """Detect and track one clip, returning rows for its track table.

    Decoding starts at :data:`~trafficflow.video.FIRST_USABLE_FRAME`, so the corrupted
    lead-in frame never reaches the detector.

    Parameters
    ----------
    model:
        A loaded model to reuse. Loading costs several seconds, so a full pass loads
        once and passes it in.

    Returns
    -------
    list of tuple
        Rows in :data:`~trafficflow.tracks.COLUMNS` order.
    """
    settings = config or DetectorConfig()
    tracker = FrameTracker(settings, model)
    rows: list[tuple] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for number, frame in iter_frames(clip.video_path(data_root)):  # skips the corrupted frame 1
            rows.extend(tracker.track(frame, number, image))
    return rows


def detect_all(
    db: TrafficDatabase,
    image: ImageGeometry,
    config: DetectorConfig | None = None,
    paths: Paths | None = None,
    clips: Sequence[str] | None = None,
    *,
    skip_existing: bool = True,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> dict[str, int]:
    """Run the pass over the whole database, writing one table per clip.

    Parameters
    ----------
    skip_existing:
        Leave clips that already have a table *made with these settings*. The pass takes
        hours, so it must be resumable after an interruption -- but a table produced by a
        different ``imgsz`` or ``confidence`` is not a resumption, it is a different
        measurement, and is re-detected.
    progress:
        Callback ``(done, total, clip_name, rows_written)``.

    Returns
    -------
    dict
        Clip name to number of rows written.
    """
    settings = config or DetectorConfig()
    where = paths or Paths()
    where.tracks.mkdir(parents=True, exist_ok=True)

    names = list(clips) if clips is not None else [clip.name for clip in db.clips]
    model = load_model(settings)
    written: dict[str, int] = {}

    for done, name in enumerate(names, start=1):
        target = where.tracks / f"{name}.csv"
        # A table only counts as present if it was produced by *these* settings. Skipping
        # on existence alone silently mixes tables from different detectors into one
        # dataset, and every average computed over them is then meaningless.
        if skip_existing and target.exists() and not stale_fields(target, settings):
            written[name] = -1  # -1 distinguishes "already present" from "found nothing"
        else:
            rows = detect_clip(db[name], db.data_root, image, settings, model=model)
            write_tracks(target, rows)
            write_settings(target, settings)
            written[name] = len(rows)
        if progress is not None:
            progress(done, len(names), name, written[name])
    return written
