"""The congestion classifier trained in Colab: its input, loading it, and running it.

The model is trained in ``02_finetune.ipynb``, which is self-contained and runs
on a GPU. What comes back is a single **ONNX** file, and the choice of format is the whole
design of this module.

**It needs nothing from the framework that trained it.** No architecture definition to
keep in sync, no torch version to match against whatever Colab shipped that month.
``onnxruntime`` reads the graph out of the file. The alternative, a TorchScript ``.pt``,
welds the two ends of that seam back together and is deprecated besides -- ``torch.jit``
already warns that it may break on Python 3.14.

**It carries its own preprocessing.** The roadway crop, the frame spacing, the input size,
the channel order and the normalisation statistics live in the file's ONNX metadata, put
there by the notebook. This module reads them out and uses them rather than declaring its
own. Preprocessing that disagrees with training is the classic way a model that scored
well in a notebook quietly returns nonsense in production, and a recipe that cannot be
separated from its model cannot disagree with it.

Model input
-----------
**The frame is cropped to the roadway.** Everything left of ``image.roadway_crop_x``
is embankment, barrier, and -- the reason the crop exists -- the opposing northbound
carriageway. Northbound congestion peaks at the opposite time of day, so a network
given the full frame could score well by reading the wrong road.

**One frame is not enough.** Congestion is density *and* speed, and *light* versus
*medium* is mostly a speed difference at comparable density. A still frame carries no
motion at all, so each sample is a short stack of frames spaced far enough apart for
movement to be visible.

Note the effective sample size is **254, not the number of stacks**. Within one 5.29 s
clip from a fixed camera the frames are near-duplicates, so drawing more stacks per
clip buys very little independence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from trafficflow.config import ImageGeometry, Paths
from trafficflow.dataset import Clip
from trafficflow.video import FIRST_USABLE_FRAME, iter_frames

__all__ = [
    "StackSpec",
    "load_clip_stacks",
    "CongestionModel",
    "Prediction",
    "load_congestion_model",
]

MODEL_NAME = "congestion_cnn.onnx"


# -- model input -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StackSpec:
    """How a clip is turned into samples.

    Attributes
    ----------
    frames_per_stack:
        Frames in one sample. Three is enough to show whether traffic is moving.
    frame_gap:
        Frames between consecutive members of a stack. At 10 fps a gap of 10 is one second,
        which the notebook measured as separating the classes twice as well as half a
        second does.
    starts:
        Offsets, in frames from the first usable frame, of each window scored. Read out of
        the model file, so a score is repeatable.
    size:
        ``(height, width)`` each frame is resized to.
    """

    frames_per_stack: int = 3
    frame_gap: int = 10
    starts: tuple[int, ...] = (0, 10, 20)
    size: tuple[int, int] = (160, 160)

    @property
    def span(self) -> int:
        """Frames from the first to the last member of one stack."""
        return (self.frames_per_stack - 1) * self.frame_gap

    def stack_starts(self, n_frames: int) -> list[int]:
        """First frame number of each window, as 1-based frame numbers.

        A clip too short for a given offset is clamped back, so a short clip yields fewer
        windows rather than reading past its end.
        """
        last_start = max(FIRST_USABLE_FRAME, n_frames - self.span)
        wanted = {min(FIRST_USABLE_FRAME + offset, last_start) for offset in self.starts}
        return sorted(wanted)

    def frame_numbers(self, n_frames: int) -> list[int]:
        """Every frame number any stack needs, sorted, so a clip is decoded once."""
        last = max(FIRST_USABLE_FRAME, n_frames)
        wanted = {
            min(start + member * self.frame_gap, last)
            for start in self.stack_starts(n_frames)
            for member in range(self.frames_per_stack)
        }
        return sorted(wanted)


def load_clip_stacks(
    clip: Clip,
    data_root,
    spec: StackSpec,
    image: ImageGeometry,
) -> np.ndarray:
    """Read one clip and return its temporal stacks.

    Returns
    -------
    numpy.ndarray
        Shape ``(stacks, frames_per_stack, height, width, 3)``, dtype ``uint8``,
        channels in BGR order as OpenCV decodes them. Cropping happens before resizing,
        so the aspect change is applied to the road, not to sky and trees.
    """
    height, width = spec.size
    wanted_set = set(spec.frame_numbers(clip.n_frames))

    collected: dict[int, np.ndarray] = {}
    for number, frame in iter_frames(clip.video_path(data_root)):
        if number in wanted_set:
            collected[number] = cv2.resize(image.crop(frame), (width, height), interpolation=cv2.INTER_AREA)
        if len(collected) == len(wanted_set):
            break

    if not collected:
        raise ValueError(f"{clip.name}: no frames could be decoded")

    # A clip may end early; clamp any missing member to the last frame we did get.
    last_available = max(collected)
    stacks = [
        np.stack([
            collected[min(start + member * spec.frame_gap, last_available)]
            for member in range(spec.frames_per_stack)
        ])
        for start in spec.stack_starts(clip.n_frames)
    ]
    return np.stack(stacks)


# -- the model -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction:
    """One predicted congestion class and the confidence behind it."""

    label: str
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


class CongestionModel:
    """A trained congestion classifier, with the preprocessing it was trained under.

    Examples
    --------
    >>> model = load_congestion_model()                      # doctest: +SKIP
    >>> model.predict(clip, data_root).label                 # doctest: +SKIP
    'heavy'
    """

    def __init__(self, session, preprocessing: dict) -> None:
        self.session = session
        self.preprocessing = preprocessing
        self.classes: list[str] = list(preprocessing["classes"])

        frames = int(preprocessing["n_frames"])
        size = int(preprocessing["size"])

        #: Sampling geometry, reconstructed from the training run rather than guessed.
        self.spec = StackSpec(
            frames_per_stack=frames,
            frame_gap=int(preprocessing["gap"]),
            starts=tuple(int(s) for s in preprocessing["test_starts"]),
            size=(size, size),
        )
        #: Only the crop matters for inference; the other fields describe the source clips.
        self.image = ImageGeometry(
            width=320,
            height=240,
            fps=10.0,
            roadway_crop_x=int(preprocessing["crop_x"]),
            has_burnt_in_timestamp=False,
        )

        # Normalisation statistics, tiled across the stacked channels, as during
        # training. Required, not defaulted: silently substituting ImageNet's figures
        # for a model normalised some other way shifts every input and degrades the
        # predictions without raising anything, which is exactly the guessed-
        # preprocessing failure this class exists to prevent. Every export written by
        # 02_finetune.ipynb carries both, so a file missing them is a broken export.
        missing = [key for key in ("mean", "std") if key not in preprocessing]
        if missing:
            raise KeyError(
                f"the model export is missing its normalisation {', '.join(missing)}; "
                "re-export from 02_finetune.ipynb rather than assuming ImageNet values"
            )
        mean = preprocessing["mean"]
        std = preprocessing["std"]
        self._mean = np.array(list(mean) * frames, dtype=np.float32).reshape(-1, 1, 1)
        self._std = np.array(list(std) * frames, dtype=np.float32).reshape(-1, 1, 1)

    def _to_input(self, stacks: np.ndarray) -> np.ndarray:
        """Apply exactly the training-time preprocessing to ``(S, F, H, W, 3)`` uint8."""
        if self.preprocessing.get("colour", "RGB").upper() == "RGB":
            stacks = stacks[..., ::-1]  # OpenCV decodes BGR
        array = np.ascontiguousarray(stacks).transpose(0, 1, 4, 2, 3)
        # (S, F, 3, H, W) -> (S, F*3, H, W), matching the widened first convolution
        array = array.reshape(array.shape[0], -1, *array.shape[3:])
        return ((array.astype(np.float32) / 255.0) - self._mean) / self._std

    def predict_stacks(self, stacks: np.ndarray) -> Prediction:
        """Classify from temporal stacks, averaging logits across them before the softmax."""
        logits = self.session.run(None, {"stacks": self._to_input(stacks)})[0].mean(axis=0)
        exponentials = np.exp(logits - logits.max())        # shifted, so it cannot overflow
        scores = dict(zip(self.classes, (exponentials / exponentials.sum()).tolist()))
        return Prediction(label=max(scores, key=scores.get), probabilities=scores)

    def predict(self, clip: Clip, data_root: Path | str) -> Prediction:
        """Classify a clip of the database, decoding it first."""
        return self.predict_stacks(load_clip_stacks(clip, data_root, self.spec, self.image))

    def prepare_frame(self, image: np.ndarray) -> np.ndarray:
        """Crop and resize one 320x240 BGR frame exactly as training did."""
        height, width = self.spec.size
        return cv2.resize(self.image.crop(image), (width, height), interpolation=cv2.INTER_AREA)

    def predict_frames(
        self, frames: dict[int, np.ndarray], latest: int, gap: int | None = None
    ) -> Prediction | None:
        """Classify the moment ending at frame ``latest``, for the live page.

        ``frames`` holds :meth:`prepare_frame` outputs by frame number. Uses frames ``gap``
        apart (one second at 10 fps) and returns ``None`` until all of them exist -- the
        first two seconds of a video.
        """
        step = gap or self.spec.frame_gap
        wanted = [latest - step * k for k in reversed(range(self.spec.frames_per_stack))]
        if any(number not in frames for number in wanted):
            return None
        return self.predict_stacks(np.stack([frames[number] for number in wanted])[None])


def load_congestion_model(path: Path | str | None = None) -> CongestionModel:
    """Load the ONNX model exported by the fine-tuning notebook.

    Raises
    ------
    FileNotFoundError
        With instructions, if the notebook has not been run and downloaded yet.
    KeyError
        If the file carries no preprocessing metadata. Guessing the preprocessing is
        exactly the failure this format was chosen to make impossible.
    """
    target = Path(path) if path else Paths().models / MODEL_NAME
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run 02_finetune.ipynb in Colab, then put the "
            f"downloaded {MODEL_NAME} in models/."
        )

    import onnxruntime  # imported here: the rest of the package does not need it

    session = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    carried = session.get_modelmeta().custom_metadata_map.get("preprocessing")
    if carried is None:
        raise KeyError(
            f"{target} carries no preprocessing metadata, so the preprocessing it was "
            f"trained under is unknown. Re-export it with the current "
            f"02_finetune.ipynb, which writes it into the file."
        )
    return CongestionModel(session, json.loads(carried))
