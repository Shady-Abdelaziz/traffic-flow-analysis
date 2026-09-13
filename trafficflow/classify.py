"""Congestion classifier built on the measured physical parameters (variant A).

The project ships two congestion models, and the comparison between them is the point.

**Variant B**, trained in ``02_finetune.ipynb``, is a fine-tuned CNN reading
pixels. It scores well but cannot say *why*: an image classifier that calls a clip *heavy*
offers no quantity a traffic engineer can act on.

**Variant A**, here, classifies from the parameters the rest of the pipeline measured --
density, speed, occupancy, lane changes. It is a small model on a handful of interpretable
numbers, and because those numbers are physical, its decisions can be read back as a rule:
*above this density and below this speed, the segment is congested.* That is a statement
about the road, not about the classifier.

Both are scored by the identical harness in :mod:`trafficflow.evaluate`, over the same
folds, against the same baselines. Different features, same yardstick.

Features deliberately exclude the recording hour: on this database it nearly determines
the label on its own, so including it would produce a good score that proves nothing
about whether the road was ever looked at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from trafficflow.dataset import TRAFFIC_CLASSES

__all__ = ["FEATURE_COLUMNS", "ParameterClassifier", "make_fit_predict"]


#: Measured quantities used as features. Every one is a physical property of the traffic,
#: so a decision made from them can be explained in engineering terms.
#:
#: Note what is absent: ``hour``, ``weather`` and ``day``. Hours 06-12 and 19-20 are
#: 100% light while 15 and 17 hold no light clips at all, and weather is confounded with
#: the hour (74 of 78 rain clips are labelled light, because it rained off-peak). A model
#: given those would score well without ever reading the road.
FEATURE_COLUMNS: tuple[str, ...] = (
    "density_pc_mi_ln",
    "vehicles_per_frame",
    "speed_median_kmh",
    "speed_p85_kmh",
    "speed_std_kmh",
    "lane_imbalance",
    "changes_per_vehicle",
    "heavy_vehicle_percent",
)


@dataclass
class ParameterClassifier:
    """A gradient-boosted classifier over the measured parameters.

    Trees rather than a linear model because the relationship is not linear: speed
    barely moves as density rises through the free-flow range and then collapses past a
    critical point. That knee is exactly what a tree split represents, and recovering
    the knee is itself one of the project's findings.
    """

    model: object | None = None
    classes: Sequence[str] = TRAFFIC_CLASSES

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> "ParameterClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier

        # Small dataset -- 254 clips, ~190 per training fold -- so the model is kept
        # small deliberately. Anything larger memorises the fold.
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.08,
            l2_regularization=1.0,
            class_weight="balanced",  # counters 165 / 45 / 44
            random_state=0,
        )
        self.model.fit(features, list(labels))
        return self

    def predict(self, features: np.ndarray) -> list[str]:
        if self.model is None:
            raise RuntimeError("classifier has not been fitted")
        return list(self.model.predict(features))


def build_feature_matrix(
    rows: dict[str, dict], names: Sequence[str]
) -> tuple[np.ndarray, list[str | None]]:
    """Assemble the feature matrix for the given clips.

    Parameters
    ----------
    rows:
        Clip name to its parameter row, as written by :mod:`trafficflow.pipeline`.
    names:
        Clips to include, in order.

    Returns
    -------
    tuple
        ``(features, labels)`` with features shaped ``(len(names), len(FEATURE_COLUMNS))``.
        ``labels`` is ``None`` at any position whose clip has no row.

    Notes
    -----
    A blank cell means the measurement was genuinely unavailable -- typically a clip
    where no track was long enough to fit a speed. It becomes NaN rather than 0, because
    a speed of zero means stopped traffic and inventing that would push the model toward
    calling the clip heavy.

    A clip with no row *at all* -- detection has not reached it yet -- is treated the
    same way: an all-NaN row rather than a dropped one. Dropping it would silently
    shorten the prediction set, and :func:`trafficflow.evaluate.cross_validate`
    rightly refuses a fold that does not answer for every clip it was asked about.
    """
    blank: dict = {}
    features = np.array(
        [[_as_float(rows.get(name, blank).get(column)) for column in FEATURE_COLUMNS]
         for name in names],
        dtype=np.float64,
    ).reshape(len(names), len(FEATURE_COLUMNS))
    labels = [rows.get(name, blank).get("traffic_class") for name in names]
    return features, labels


def _as_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def make_fit_predict(rows: dict[str, dict]):
    """Build a ``fit_predict`` for :func:`trafficflow.evaluate.cross_validate`.

    Wrapping it this way is what makes the comparison with the CNN fair: both models
    reach the scoreboard through the same function, the same folds and the same metrics.

    Training drops clips with no measured row, because a training example needs a
    label. Testing does not: every clip the fold asks about is predicted, from an
    all-NaN feature row if that is all there is. HistGradientBoosting takes NaN
    natively, so a clip detection has not reached yet gets the model's prior rather
    than no answer -- which is what a half-finished ``run.py detect`` produces, and
    what used to abort ``run.py train`` in :func:`trafficflow.evaluate.cross_validate`.
    """

    def fit_predict(train: Sequence[str], test: Sequence[str]) -> dict[str, str]:
        train_names = [name for name in train if name in rows]
        test_names = list(test)

        train_features, _ = build_feature_matrix(rows, train_names)
        test_features, _ = build_feature_matrix(rows, test_names)
        train_labels = [rows[name]["traffic_class"] for name in train_names]

        classifier = ParameterClassifier().fit(train_features, train_labels)
        predicted = classifier.predict(test_features)
        return dict(zip(test_names, predicted))

    return fit_predict
