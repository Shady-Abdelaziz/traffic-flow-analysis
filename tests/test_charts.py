"""Tests for the fundamental diagram fit.

This fit is the headline finding, and until now nothing tested it. The two things that
can go wrong are both silent: a unit mismatch produces a capacity that looks plausible
next to the HCM's 2300 veh/h/ln while being wrong by a constant factor, and a slope that
does not come out negative produces infinities that travel a long way before anything
notices. Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from trafficflow.charts import KM_PER_MILE, fit_fundamental_diagram


def greenshields(free_flow_kmh: float, jam_density: float, n: int = 40) -> tuple:
    """Density and speed lying exactly on ``v = v_free * (1 - k / k_jam)``."""
    density = np.linspace(1.0, jam_density * 0.9, n)
    speed = free_flow_kmh * (1.0 - density / jam_density)
    return density, speed


# -- what the fit recovers -------------------------------------------------------


def test_the_fit_recovers_the_curve_it_was_given():
    density, speed = greenshields(100.0, 120.0)
    fit = fit_fundamental_diagram(density, speed)

    assert fit.free_flow_speed_kmh == pytest.approx(100.0, rel=1e-6)
    assert fit.jam_density == pytest.approx(120.0, rel=1e-6)
    assert fit.critical_density == pytest.approx(60.0, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.n_points == 40


def test_capacity_is_a_flow_not_a_unit_salad():
    """Density is per *mile*; speed is in km/h. One of them has to be converted.

    Multiplying them as they stand gives pc-km per mile-hour, which is not a flow, and
    is larger than the real figure by exactly the number of kilometres in a mile. That
    factor is the whole point of this test: the capacity is compared against the HCM's
    2300 veh/h/ln in two recommendations, and a 1.609x error there is an argument for
    doing nothing to a road that needs attention.
    """
    density, speed = greenshields(100.0, 120.0)
    fit = fit_fundamental_diagram(density, speed)

    # Worked by hand: critical density 60 pc/mi/ln, speed there 50 km/h = 31.07 mph.
    expected = 60.0 * (50.0 / KM_PER_MILE)
    assert fit.capacity_veh_per_h_per_ln == pytest.approx(expected, rel=1e-9)
    assert fit.capacity_veh_per_h_per_ln == pytest.approx(1864.5, abs=0.5)

    # And it is *not* the unconverted product, which is what it used to be.
    assert fit.capacity_veh_per_h_per_ln != pytest.approx(60.0 * 50.0, rel=1e-3)


# -- refusing to answer ----------------------------------------------------------


@pytest.mark.parametrize("slope", [0.5, 0.0], ids=["rising", "flat"])
def test_a_relationship_that_does_not_fall_has_no_capacity(slope):
    """Speed that does not fall with density is not Greenshields, and has no jam density.

    Returning ``inf`` here used to look harmless. It is not: the value reaches the
    website through JSON, which has no way to encode an infinity, so the Results
    endpoint failed as a whole rather than as one missing tile. A flat relationship is
    the boundary of the same failure -- the slope is the denominator, so zero is where
    it divides rather than merely pointing the wrong way.
    """
    density = np.linspace(10.0, 80.0, 20)
    speed = 40.0 + slope * density
    with pytest.raises(ValueError, match="does not fall with density"):
        fit_fundamental_diagram(density, speed)


def test_too_few_usable_clips_is_refused_with_a_count():
    density = np.array([10.0, np.nan, 30.0])
    speed = np.array([90.0, 80.0, np.nan])
    with pytest.raises(ValueError, match="got 1"):
        fit_fundamental_diagram(density, speed)


def test_clips_missing_either_measurement_are_dropped_not_zeroed():
    """A blank speed means unmeasured. Treating it as 0 would fake a jam."""
    density, speed = greenshields(100.0, 120.0, n=20)
    density = np.append(density, [55.0, 60.0])
    speed = np.append(speed, [np.nan, np.nan])

    fit = fit_fundamental_diagram(density, speed)
    assert fit.n_points == 20
    assert fit.free_flow_speed_kmh == pytest.approx(100.0, rel=1e-6)
