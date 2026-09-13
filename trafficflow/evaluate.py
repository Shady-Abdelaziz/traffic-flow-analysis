"""The evaluation protocol, and the baselines a real model has to clear.

Everything scored in this project goes through :func:`cross_validate` -- the parameter
classifier, the fine-tuned network, and the baselines. They differ only in the
``fit_predict`` function handed to it, so no variant can accidentally be measured on a
different split, a different metric, or a different definition of "correct".

**Which split.** Consecutive clips are consecutive *recordings*: ``...x01638`` and
``...x01639`` are five seconds apart, the same vehicles a few metres on. A split that
puts them on opposite sides scores a clip against its own near-duplicate.

The 254 clips form 14 separate runs, so keeping each one whole is possible.
:func:`grouped_folds` does it with ``StratifiedGroupKFold``, which keeps every recording
run in one fold while still spreading the three classes across the folds. It is the only
split used here.

**Accuracy hides a missing class.** The labels are 165 light / 45 medium / 44 heavy, so
a model that never once predicts *medium* still reaches 65%. Results therefore carry
macro-F1 and a confusion matrix alongside accuracy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn import metrics as skmetrics
from sklearn.model_selection import StratifiedGroupKFold

from trafficflow.dataset import TRAFFIC_CLASSES, Fold, TrafficDatabase

__all__ = [
    "FitPredict",
    "FoldResult",
    "CrossValidationResult",
    "confusion_matrix",
    "accuracy",
    "per_class_f1",
    "macro_f1",
    "recording_runs",
    "grouped_folds",
    "cross_validate",
    "majority_baseline",
    "nearest_recording_baseline",
]


#: A model, expressed as the only thing the protocol needs from it: given the training
#: clip names and the test clip names, return a predicted class for each test clip.
#: Everything else -- feature extraction, architecture, hyperparameters -- is the
#: model's own business.
FitPredict = Callable[[Sequence[str], Sequence[str]], Mapping[str, str]]


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = TRAFFIC_CLASSES,
) -> np.ndarray:
    """Counts of true class (rows) against predicted class (columns)."""
    return skmetrics.confusion_matrix(y_true, y_pred, labels=list(labels))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction correct. Reported, but never on its own -- see the module docstring."""
    return float(skmetrics.accuracy_score(y_true, y_pred))


def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = TRAFFIC_CLASSES,
) -> dict[str, float]:
    """F1 for each class. A class the model never predicts scores 0, not NaN."""
    scores = skmetrics.f1_score(
        y_true, y_pred, labels=list(labels), average=None, zero_division=0
    )
    return {label: float(score) for label, score in zip(labels, scores)}


def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = TRAFFIC_CLASSES,
) -> float:
    """Unweighted mean of the per-class F1 scores.

    Unweighted is the point: at 165/45/44 a weighted average lets a model coast on
    *light* alone, which is exactly the failure this number exists to catch.
    """
    return float(
        skmetrics.f1_score(
            y_true, y_pred, labels=list(labels), average="macro", zero_division=0
        )
    )


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Scores for one fold of the protocol."""

    fold: int
    y_true: tuple[str, ...]
    y_pred: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        return accuracy(self.y_true, self.y_pred)

    @property
    def macro_f1(self) -> float:
        return macro_f1(self.y_true, self.y_pred, self.labels)

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, self.labels)


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Scores pooled across every fold.

    Pooled predictions, not averaged fold scores: the folds are complements covering
    all 254 clips exactly once, so pooling gives each clip equal weight and yields one
    confusion matrix over the whole database.
    """

    name: str
    folds: tuple[FoldResult, ...]
    labels: tuple[str, ...]

    @property
    def y_true(self) -> tuple[str, ...]:
        return tuple(label for fold in self.folds for label in fold.y_true)

    @property
    def y_pred(self) -> tuple[str, ...]:
        return tuple(label for fold in self.folds for label in fold.y_pred)

    @property
    def accuracy(self) -> float:
        return accuracy(self.y_true, self.y_pred)

    @property
    def macro_f1(self) -> float:
        return macro_f1(self.y_true, self.y_pred, self.labels)

    @property
    def per_class_f1(self) -> dict[str, float]:
        return per_class_f1(self.y_true, self.y_pred, self.labels)

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, self.labels)

    @property
    def fold_accuracies(self) -> tuple[float, ...]:
        return tuple(fold.accuracy for fold in self.folds)

    def predicted_classes(self) -> set[str]:
        """Classes the model actually emitted -- if one is missing, say so loudly."""
        return set(self.y_pred)

    def to_dict(self) -> dict:
        """Plain-Python summary, ready for ``json.dump``."""
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "fold_accuracies": [round(a, 4) for a in self.fold_accuracies],
            "labels": list(self.labels),
            "confusion_matrix": self.confusion.tolist(),
            "classes_never_predicted": sorted(set(self.labels) - self.predicted_classes()),
        }

    def summary(self) -> str:
        """Human-readable report, including the confusion matrix."""
        lines = [
            f"{self.name}",
            f"  accuracy  {self.accuracy:6.1%}",
            f"  macro-F1  {self.macro_f1:6.3f}",
            "  per-class F1: "
            + "  ".join(f"{k} {v:.3f}" for k, v in self.per_class_f1.items()),
            "  confusion (rows = true, cols = predicted):",
            "            " + "".join(f"{label:>9}" for label in self.labels),
        ]
        for label, row in zip(self.labels, self.confusion):
            lines.append(f"    {label:>8}" + "".join(f"{count:9d}" for count in row))
        missing = sorted(set(self.labels) - self.predicted_classes())
        if missing:
            lines.append(f"  WARNING never predicted: {', '.join(missing)}")
        return "\n".join(lines)


def recording_runs(db: TrafficDatabase) -> dict[str, int]:
    """Map each clip to the continuous recording it belongs to.

    A run is an unbroken stretch of serials on one day. The 254 clips form 14 of them,
    which is what makes a leak-free split possible at all: if they were one unbroken
    run, no split could keep near-duplicates on the same side.
    """
    ordered = sorted(db, key=lambda clip: (clip.date, clip.serial))
    runs: dict[str, int] = {}
    current, previous = 0, None
    for clip in ordered:
        if previous is not None and not (
            clip.date == previous.date and clip.serial == previous.serial + 1
        ):
            current += 1
        runs[clip.name] = current
        previous = clip
    return runs


def grouped_folds(
    db: TrafficDatabase, *, n_splits: int = 4, seed: int = 0
) -> tuple[Fold, ...]:
    """Four folds that never split a recording across training and test.

    Stratified as well as grouped: grouping alone left a fold holding two *heavy*
    clips, and a macro-F1 computed on two clips is noise. Stratifying keeps every
    fold at seven or more of each class.
    """
    names = np.array([clip.name for clip in db])
    truth = db.labels()
    y = np.array([truth[name] for name in names])
    groups = np.array([recording_runs(db)[name] for name in names])

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return tuple(
        Fold(index=index, train=tuple(names[train]), test=tuple(names[test]))
        for index, (train, test) in enumerate(splitter.split(names, y, groups))
    )


def cross_validate(
    db: TrafficDatabase,
    fit_predict: FitPredict,
    *,
    name: str,
    labels: Sequence[str] = TRAFFIC_CLASSES,
) -> CrossValidationResult:
    """Run ``fit_predict`` over the :func:`grouped_folds` and pool the predictions.

    Parameters
    ----------
    db:
        The loaded database, supplying both the folds and the ground-truth labels.
    fit_predict:
        Called once per fold as ``fit_predict(train_names, test_names)`` and must
        return a class for every test clip. It receives clip *names*, never indices,
        so there is no row order to get out of step.
    name:
        Label for the result, e.g. ``"majority-class baseline"``.

    Raises
    ------
    ValueError
        If a prediction is missing for some test clip, or a predicted class is not in
        ``labels``.
    """
    truth = db.labels()
    results: list[FoldResult] = []

    for fold in grouped_folds(db):
        predictions = fit_predict(fold.train, fold.test)

        missing = [clip for clip in fold.test if clip not in predictions]
        if missing:
            raise ValueError(
                f"fold {fold.index}: no prediction for {len(missing)} clip(s), "
                f"first is {missing[0]}"
            )
        unknown = {p for p in predictions.values() if p not in labels}
        if unknown:
            raise ValueError(f"fold {fold.index}: predicted unknown class(es) {unknown}")

        results.append(
            FoldResult(
                fold=fold.index,
                y_true=tuple(truth[clip] for clip in fold.test),
                y_pred=tuple(predictions[clip] for clip in fold.test),
                labels=tuple(labels),
            )
        )

    return CrossValidationResult(name=name, folds=tuple(results), labels=tuple(labels))


# -- baselines -----------------------------------------------------------------
#
# Both are ordinary fit_predict functions, so they are scored by exactly the same
# harness as the real models. A baseline measured differently is not a baseline.


def majority_baseline(db: TrafficDatabase) -> CrossValidationResult:
    """Always predict the most common class in the training fold. Scores 65.0%.

    The floor, not the bar: it never predicts *medium* or *heavy* at all, which is why
    macro-F1 rather than accuracy is the headline everywhere in this project.
    """

    def fit_predict(train: Sequence[str], test: Sequence[str]) -> dict[str, str]:
        truth = db.labels()
        most_common = Counter(truth[clip] for clip in train).most_common(1)[0][0]
        return {clip: most_common for clip in test}

    return cross_validate(db, fit_predict, name="majority-class baseline")


def nearest_recording_baseline(db: TrafficDatabase) -> CrossValidationResult:
    """Copy the label of the nearest clip in recording order. Reads not one pixel.

    The bar a real model has to clear: recording time alone carries part of the label,
    so on :func:`grouped_folds` this still scores 0.584 macro-F1.
    """

    def fit_predict(train: Sequence[str], test: Sequence[str]) -> dict[str, str]:
        truth = db.labels()
        # Sorted, so that when the recordings on both sides are in the training fold
        # the tie goes to the earlier one rather than to whatever the caller's list
        # order happened to be. A deployed system holds the past, not the future.
        pool = sorted(
            (db[clip].date, db[clip].serial, truth[clip]) for clip in train
        )
        predictions = {}
        for clip in test:
            date, serial = db[clip].date, db[clip].serial
            # Same day first, then closest in recording order.
            nearest = min(pool, key=lambda row: (row[0] != date, abs(row[1] - serial)))
            predictions[clip] = nearest[2]
        return predictions

    return cross_validate(db, fit_predict, name="nearest-recording probe (no pixels)")
