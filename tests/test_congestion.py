"""Tests for the congestion classifier trained in Colab.

Two different things are checked here, and only one of them needs the 45 MB model file.

**The scores.** ``eval_finetune.json`` carries the numbers quoted in the README
and in the final report, on the split that keeps each recording whole. The confusion
matrix is the raw evidence, so accuracy and macro-F1 are recomputed from it with
scikit-learn and compared against
what the notebook reported. This catches the failure that matters most: a report and a
model that came from different runs. It needs no torch and no model file.

**The loader.** That the ONNX file opens, that the preprocessing welded inside it agrees
with the copy the notebook wrote alongside, and that a prediction is a well-formed
probability distribution. Those two copies of the preprocessing are written from the same
run, so a disagreement means the model and the report came from different ones. Skipped if
the model has not been downloaded from Colab yet.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from artefacts import MODEL_PATH, REPORT_PATH, needs_model, needs_report

CLASSES = ["light", "medium", "heavy"]
#: The notebook scores every model on one split, which keeps each recording run in a
#: single fold so adjacent recordings of the same traffic never straddle train and test.
HEADLINE = "grouped by recording"


@pytest.fixture
def report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def labels_from(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand a confusion matrix back into ``(y_true, y_pred)``.

    The matrix is the raw evidence in the report. Expanding it and handing the result
    to scikit-learn is how the reported numbers get recomputed rather than trusted,
    without a second hand-written copy of the F1 formula that could disagree with the
    one the notebook used.
    """
    true = np.repeat(np.arange(len(confusion)), confusion.sum(axis=1))
    predicted = np.concatenate([np.repeat(np.arange(len(row)), row) for row in confusion])
    return true, predicted


def score(report: dict, split: str, name: str) -> dict:
    """The one reported row for a model on a split."""
    return next(
        result for result in report["results"]
        if result["split"] == split and name in result["name"]
    )


# -- the reported scores ---------------------------------------------------------


@needs_report
def test_the_confusion_matrix_covers_the_whole_database(report):
    confusion = np.array(report["confusion"][HEADLINE])
    assert confusion.shape == (3, 3)
    assert confusion.sum() == 254
    # info.txt's class balance: however the folds are drawn, cross_val_predict predicts
    # every clip exactly once.
    assert confusion.sum(axis=1).tolist() == [165, 45, 44]


@needs_report
def test_the_reported_scores_match_the_confusion_matrix(report):
    """Both reported numbers, recomputed from the evidence rather than trusted.

    This catches the failure that matters most: a report and a model that came out of
    two different runs.
    """
    true, predicted = labels_from(np.array(report["confusion"][HEADLINE]))
    reported = score(report, HEADLINE, "resnet")

    assert accuracy_score(true, predicted) == pytest.approx(reported["accuracy"], abs=1e-6)
    assert f1_score(true, predicted, average="macro") == pytest.approx(
        reported["macro_f1"], abs=1e-6
    )


@needs_report
def test_the_headline_is_the_grouped_split(report):
    """Every number quoted in the README and the report is the grouped one."""
    assert report["headline"] == HEADLINE


@needs_report
def test_the_model_beats_the_majority_floor_on_macro_f1(report):
    """Accuracy is not the test -- macro-F1 against a model that predicts one class.

    At 165/45/44 a classifier that says *light* every time reaches 65% accuracy while
    scoring zero on two classes out of three. Clearing it is the minimum claim.
    """
    model = score(report, HEADLINE, "resnet")
    majority = score(report, HEADLINE, "majority")

    assert model["macro_f1"] > majority["macro_f1"], (
        f"the network scores {model['macro_f1']:.3f} macro-F1 against the "
        f"majority floor's {majority['macro_f1']:.3f}"
    )


@needs_report
def test_the_model_beats_the_probe_that_reads_no_pixels(report):
    """The claim the whole project rests on: the pixels are doing the work.

    The probe copies the label of the nearest recording and opens no image at all. On
    the grouped split it is the honest floor, and the margin over it is the model's real
    contribution.
    """
    margin = (score(report, HEADLINE, "resnet")["macro_f1"]
              - score(report, HEADLINE, "nearest recording")["macro_f1"])
    assert margin > 0.1, margin


@needs_report
def test_the_preprocessing_the_model_needs_is_recorded(report):
    """Preprocessing that disagrees with training is how a good notebook score turns
    into nonsense locally, so the constants travel with the artefact."""
    preprocessing = report["preprocessing"]
    assert list(preprocessing["classes"]) == CLASSES
    assert preprocessing["crop_x"] == 112      # the northbound carriageway is excluded
    assert preprocessing["first_frame"] == 2   # frame 1 carries another video signal
    assert preprocessing["n_frames"] >= 2, "a single frame cannot show motion"
    assert preprocessing["colour"].upper() in {"RGB", "BGR"}


# -- the loader ------------------------------------------------------------------


@needs_model
@needs_report
def test_the_model_loads_and_agrees_with_its_recorded_preprocessing(report):
    """The model reads its preprocessing out of its own metadata, and the report carries
    a separate copy. They are written together, so disagreement means different runs."""
    from trafficflow.congestion import load_congestion_model

    model = load_congestion_model()
    preprocessing = report["preprocessing"]

    assert model.classes == CLASSES
    assert model.spec.frames_per_stack == preprocessing["n_frames"]
    assert model.spec.frame_gap == preprocessing["gap"]
    assert model.spec.size == (preprocessing["size"], preprocessing["size"])
    assert model.image.roadway_crop_x == preprocessing["crop_x"]


@needs_model
@needs_report
def test_a_prediction_is_a_probability_distribution_over_the_three_classes(report):
    from trafficflow.congestion import load_congestion_model

    model = load_congestion_model()
    preprocessing = report["preprocessing"]
    size = preprocessing["size"]

    stacks = np.zeros((2, preprocessing["n_frames"], size, size, 3), dtype=np.uint8)
    prediction = model.predict_stacks(stacks)

    assert prediction.label in CLASSES
    assert set(prediction.probabilities) == set(CLASSES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert prediction.confidence == max(prediction.probabilities.values())


@needs_model
def test_live_prediction_waits_for_three_frames_one_second_apart():
    from trafficflow.congestion import load_congestion_model

    model = load_congestion_model()
    blank = model.prepare_frame(np.zeros((240, 320, 3), dtype=np.uint8))
    assert blank.shape == (*model.spec.size, 3)

    frames = {n: blank for n in range(2, 22)}                   # frames 2..21
    assert model.predict_frames(frames, latest=21) is None      # would need frame 1
    frames[22] = blank
    prediction = model.predict_frames(frames, latest=22)        # 2, 12, 22
    assert prediction is not None and prediction.label in CLASSES


def test_a_missing_model_says_how_to_produce_it():
    """Needs no model, and no onnxruntime: the path is checked before either is touched."""
    from trafficflow.congestion import load_congestion_model

    with pytest.raises(FileNotFoundError, match="02_finetune"):
        load_congestion_model(MODEL_PATH.parent / "does_not_exist.onnx")
