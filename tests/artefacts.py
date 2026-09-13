"""Skip conditions for the files the notebooks produce.

Four modules each decided separately whether ``calibration.yaml``, the ONNX model or
``eval_finetune.json`` was present, and worded the reason differently. They are all
outputs of a Colab notebook rather than checked-in source, so a fresh clone has none of
them: their absence is a state to skip on, not to fail on. Collecting the conditions
here keeps that judgement -- and the instruction for how to produce each file -- in one
place, and keeps ``conftest.py`` to fixtures.
"""

from __future__ import annotations

import pytest

from trafficflow.config import Paths

CALIBRATION_PATH = Paths().calibration
MODEL_PATH = Paths().models / "congestion_cnn.onnx"
REPORT_PATH = Paths().models / "eval_finetune.json"

#: Where the bottom of the frame lands on the road, and how many metres a pixel covers,
#: are properties of this camera rather than of any synthetic fixture.
needs_calibration = pytest.mark.skipif(
    not CALIBRATION_PATH.exists(),
    reason="calibration.yaml not present -- run 01_calibration.ipynb",
)

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"{MODEL_PATH.name} not present -- run 02_finetune.ipynb",
)

needs_report = pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason=f"{REPORT_PATH.name} not present -- run 02_finetune.ipynb",
)
