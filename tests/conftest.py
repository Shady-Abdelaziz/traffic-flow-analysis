"""Fixtures shared across the suite.

The identity-homography road frame used to be written out separately in five modules.
It lives in ``synthetic_road`` now, exposed here as a fixture so every module that
measures on the synthetic road measures on the same one.

Skip conditions for the notebook-produced files are in ``artefacts``.
"""

from __future__ import annotations

import pytest

from synthetic_road import identity_road_frame


@pytest.fixture
def frame():
    """The synthetic road: identity homography, so one pixel is one metre."""
    return identity_road_frame()
