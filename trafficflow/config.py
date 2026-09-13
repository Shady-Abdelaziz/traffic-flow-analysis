"""Typed access to the two configuration files and the project's directory layout.

Configuration is split in two because the halves have opposite lifecycles:

``config/site.yaml``
    Hand-written facts about the road, each carrying the primary source it came
    from. Code only ever reads this. Writing it from code would destroy the source
    comments, which are the reason it exists.
``config/calibration.yaml``
    Produced by ``01_calibration.ipynb`` and overwritten wholesale each
    time calibration runs. Never hand-edited.

Reading configuration through the dataclasses here rather than raw dictionaries means
a renamed key fails immediately with a clear error, instead of surfacing later as a
``KeyError`` in the middle of an hour-long detection pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "PROJECT_ROOT",
    "Paths",
    "Roadway",
    "ImageGeometry",
    "LevelOfService",
    "SiteConfig",
    "Calibration",
    "load_site",
    "load_calibration",
]


#: Repository root, i.e. the directory holding ``config/`` and ``trafficflow/``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Paths:
    """Where each stage reads from and writes to.

    Stages communicate through these directories rather than by calling each other,
    so re-rendering a figure never re-runs detection.
    """

    root: Path = PROJECT_ROOT

    @property
    def archive(self) -> Path:
        """The dataset as shipped."""
        return self.root / "archive"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def models(self) -> Path:
        """Every model file: YOLO checkpoints, the congestion CNN and its scores."""
        return self.root / "models"

    @property
    def calibration(self) -> Path:
        """The calibration written by 01_calibration.ipynb, beside site.yaml."""
        return self.config / "calibration.yaml"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def tracks(self) -> Path:
        """One track table per clip, written by the detection stage."""
        return self.output / "tracks"

    @property
    def uploads(self) -> Path:
        """One folder per video uploaded to the website: its video, tracks and metadata.

        Separate from :attr:`tracks`, which is keyed by dataset clip name. An upload has
        no dataset record, so mixing the two would collide on names and would make
        "clear the dataset but keep my uploads" impossible to express.
        """
        return self.output / "uploads"

    @property
    def parameters(self) -> Path:
        return self.output / "parameters"

    @property
    def figures(self) -> Path:
        return self.output / "figures"

    @property
    def videos(self) -> Path:
        return self.output / "videos"

    @property
    def reports(self) -> Path:
        return self.output / "reports"

    def ensure(self) -> "Paths":
        """Create every output directory that does not yet exist."""
        for directory in (
            self.tracks,
            self.uploads,
            self.parameters,
            self.figures,
            self.videos,
            self.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class Roadway:
    """What is left of the hand-written road facts.

    Everything geometric now comes from :class:`Calibration`, measured from the
    footage. These two survive because neither can be: the lane count is not
    countable when the interior lane lines are invisible, and the posted limit is
    not a property of the image at all.
    """

    lane_count: int
    speed_limit_kmh: float


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    """Frame dimensions and the roadway crop, in native 320x240 pixels."""

    width: int
    height: int
    fps: float
    roadway_crop_x: int
    has_burnt_in_timestamp: bool

    def crop(self, image: Any) -> Any:
        """Return the roadway region of ``image``, excluding the northbound carriageway.

        Applied to every frame that reaches a model. Cropping is the only defence
        against a classifier keying on opposing traffic, whose peak runs at the
        opposite time of day.
        """
        return image[:, self.roadway_crop_x :]


@dataclass(frozen=True, slots=True)
class LevelOfService:
    """Highway Capacity Manual thresholds for a basic freeway segment."""

    density_bounds_pc_per_mi_per_ln: dict[str, float]
    capacity_veh_per_h_per_ln: float

    def grade(self, density_pc_per_mi_per_ln: float) -> str:
        """Convert a density to its level-of-service letter, A (free) to F (breakdown)."""
        for letter, upper in sorted(
            self.density_bounds_pc_per_mi_per_ln.items(), key=lambda kv: kv[1]
        ):
            if density_pc_per_mi_per_ln <= upper:
                return letter
        return "F"


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Everything in ``config/site.yaml``."""

    location: dict[str, Any]
    roadway: Roadway
    image: ImageGeometry
    level_of_service: LevelOfService


@dataclass(frozen=True, slots=True)
class Calibration:
    """Everything in ``calibration.yaml``, written by the calibration notebook.

    Attributes
    ----------
    homography:
        3x3 matrix mapping image points to the road plane in metres. This is the only
        field the pipeline actually computes with; the rest are the audit trail.
    camera_height_m:
        Perpendicular height of the camera above the road surface, taken from the
        plane fitted to UniDepth's metric depth. Its scale is the model's, not a
        survey's -- see the note on ``measured_width_m``.
    horizon_row:
        Image row of the road's vanishing point, ``y0`` -- where the fitted plane's
        inverse depth reaches zero, which is to say infinitely far away.
    focal_length_px:
        UniDepth's own predicted ``f`` -- the one ``plane_coefficients`` were fitted
        with. An audit number, not an input: nothing in this package measures with it,
        because every distance is read off the plane instead. Recording the median of
        the estimates here would pair a camera height fitted with one focal length
        against a different one.
    focal_estimates:
        Every model's answer -- UniDepth's predicted intrinsics, and GeoCalib's --
        kept so the spread is auditable rather than hidden behind a single number.
    scale_spread:
        How far the second opinion sits from ``focal_length_px``. GeoCalib's only job
        is to disagree; above 0.15 it has disagreed enough that the calibration is not
        trustworthy. Recorded for the audit trail; nothing reads it at run time.
    left_edge_poly:
        Coefficients of a quadratic giving the image column of the yellow left-edge
        line at each image row, highest power first.

        **The road curves.** Over the visible stretch the left edge sweeps sideways
        by an appreciable fraction of a lane -- the notebook prints the figure and
        records it in ``notes`` -- far too much to treat as straight. So lateral
        position is measured from this line rather than from a fixed offset, and the
        homography's own x output is only the distance from the camera axis.
    principal_point_x:
        Image column of the optical axis, from the predicted intrinsics.
    measured_width_m:
        Width of the travelled way as measured from the footage, between the two
        outermost traced markings.

        Nothing is compared against it, because there is nothing valid to compare it
        with: the WSDOT State Highway Log is a 2024 document and the live camera at
        this milepost is a different camera at a different zoom, while this footage is
        from August 2004. It is recorded so the number the calibration implies is at
        least visible, and because its *constancy along the road* is the check that
        the horizon is right.
    plane_coefficients:
        ``[a, b, c]`` of the road plane, written as ``1/Z = a*u + b*v + c``. Every
        other geometric field here is derived from these three numbers.
    width_drift:
        How much ``measured_width_m`` varies along the road, as a fraction of itself.
        The calibration's strongest internal check, recorded rather than printed and
        forgotten: a real highway segment does not change width, so anything above a
        few percent means the horizon or one of the traced edges is wrong. Nothing can
        gate on a number that only ever appeared in a notebook's output.
    visible_road_m:
        Nearest and furthest distance along the road covered by the traced markings.
        The measurement zone reaches to the far end of it. Its near end is **not** taken
        from here: the tracing starts wherever it starts, so the zone would begin a
        metre or so inside the picture and drop the nearest vehicles. It begins at the
        bottom row of the frame instead -- see :func:`~trafficflow.pipeline.zone_bounds_m`.

        Measured **along the road plane**, in the same coordinate
        :meth:`RoadFrame.in_zone` compares against -- not depth along the optical
        axis, which is a different number by the cosine of the camera's depression.
    grade_deg, bank_deg:
        Tilt of the fitted plane along and across the road. Recorded because the road
        frame is built from the plane itself rather than from height and horizon
        alone, which is what lets a graded, banked carriageway still come out
        straight and true to scale.
    day_shift_px:
        Pixel offset from day 1 to day 2, measured by phase correlation. The camera
        moved between the two recording days.
    source, notes:
        Where the file came from and what was actually measured, written by the
        notebook so a calibration can be read months later without it to hand.
    """

    homography: list[list[float]]
    camera_height_m: float
    horizon_row: float
    focal_length_px: float
    focal_estimates: dict[str, float]
    scale_spread: float
    left_edge_poly: list[float]
    principal_point_x: float
    day_shift_px: dict[str, float]
    measured_width_m: float | None = None
    plane_coefficients: list[float] | None = None
    visible_road_m: list[float] | None = None
    width_drift: float | None = None
    grade_deg: float | None = None
    bank_deg: float | None = None
    source: str = ""
    notes: str = ""

    def left_edge_x(self, row: float) -> float:
        """Image column of the yellow left-edge line at ``row``."""
        a, b, c = self.left_edge_poly
        return a * row * row + b * row + c


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing configuration file: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_site(path: Path | str | None = None) -> SiteConfig:
    """Read ``config/site.yaml`` into typed objects.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    KeyError
        If a required key is absent -- raised here, at startup, rather than deep in a
        later stage.
    """
    raw = _load_yaml(Path(path) if path else Paths().config / "site.yaml")
    return SiteConfig(
        location=raw["location"],
        roadway=Roadway(**raw["roadway"]),
        image=ImageGeometry(**raw["image"]),
        level_of_service=LevelOfService(**raw["level_of_service"]),
    )


def load_calibration(path: Path | str | None = None) -> Calibration:
    """Read ``calibration.yaml``.

    Raises
    ------
    FileNotFoundError
        If calibration has not been run yet. The message says how to produce it.
    """
    target = Path(path) if path else Paths().calibration
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist yet -- run 01_calibration.ipynb "
            f"to produce it before any stage that converts pixels to metres"
        )
    return Calibration(**_load_yaml(target))
