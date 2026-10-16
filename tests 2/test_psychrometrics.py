"""Unit tests for the psychrometric helpers (fork feature).

The numbers here are anchored to published boundary cases, not to whatever the
implementation happens to return, so a regression in the formulas fails loudly:

* DIN 4108-2 section 6.2 mould check: 20 degC / 50 % indoors, -5 degC outdoors,
  R_si = 0.25 m2K/W gives a required surface temperature of 12.6 degC and
  f_Rsi = 0.70.
* ISO 13788 / BS 5250:2011 critical surface humidity: 80 % RH.
* Measured site data (Bubuieci, 16.10): outdoor 7.3 degC / 86 %, house
  23.3 degC / 66 %, hand-computed with the Magnus/Sonntag coefficients.
"""

from __future__ import annotations

import math

import pytest

from custom_components.humidity_intelligence.helpers.psychrometrics import (
    CRITICAL_SURFACE_RH,
    absolute_humidity,
    dew_point,
    max_safe_indoor_rh,
    moisture_excess,
    saturation_pressure,
    surface_relative_humidity,
    surface_temperature,
    vapour_pressure,
)


def test_saturation_pressure_matches_magnus_reference_points():
    # Hand-computed from e_s = 6.112 * exp(17.62 t / (243.12 + t)).
    # Note: other approximations disagree in the third digit -- Alduchov-Eskridge
    # gives 23.334 hPa and Buck 23.383 hPa at 20 degC against Magnus/Sonntag's
    # 23.326. That spread (<0.25 %) is irrelevant for control decisions, but the
    # reference values here must come from the Magnus coefficients this module
    # actually uses, otherwise the test fails for the wrong reason.
    assert saturation_pressure(0.0) == pytest.approx(6.112, abs=0.001)
    assert saturation_pressure(7.3) == pytest.approx(10.215, abs=0.01)
    assert saturation_pressure(23.3) == pytest.approx(28.538, abs=0.01)
    assert saturation_pressure(20.0) == pytest.approx(23.326, abs=0.01)


def test_vapour_pressure_scales_with_relative_humidity():
    e_s = saturation_pressure(23.3)
    assert vapour_pressure(23.3, 100.0) == pytest.approx(e_s, abs=1e-9)
    assert vapour_pressure(23.3, 66.0) == pytest.approx(0.66 * e_s, abs=1e-9)
    assert vapour_pressure(23.3, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_absolute_humidity_site_measurements():
    # Outdoor air at 86 % RH carries half the water of indoor air at 66 % RH.
    assert absolute_humidity(7.3, 86.0) == pytest.approx(6.79, abs=0.02)
    assert absolute_humidity(23.3, 66.0) == pytest.approx(13.77, abs=0.02)


def test_absolute_humidity_is_the_quantity_rh_hides():
    """Same RH, different temperature, very different amount of water."""
    cold = absolute_humidity(7.3, 86.0)
    warm = absolute_humidity(23.3, 86.0)
    assert warm > 2 * cold


def test_dew_point_site_measurements():
    assert dew_point(7.3, 86.0) == pytest.approx(5.11, abs=0.05)
    assert dew_point(23.3, 66.0) == pytest.approx(16.59, abs=0.05)
    # Boys' room: the worst dew point in the house.
    assert dew_point(23.5, 70.0) == pytest.approx(17.71, abs=0.05)


def test_dew_point_equals_temperature_at_saturation():
    assert dew_point(18.0, 100.0) == pytest.approx(18.0, abs=0.01)


def test_moisture_excess_positive_means_ventilation_dries():
    excess = moisture_excess(23.3, 66.0, 7.3, 86.0)
    assert excess == pytest.approx(6.98, abs=0.05)
    assert excess > 0


def test_moisture_excess_negative_when_outdoor_air_is_wetter():
    """A warm humid evening: RH-based logic would ventilate and import water."""
    excess = moisture_excess(24.0, 62.0, 18.0, 95.0)
    assert excess == pytest.approx(-1.10, abs=0.05)
    assert excess < 0


def test_moisture_excess_sign_flip_temperature():
    """At 86 % outdoors the drying benefit disappears near 18.7 degC."""
    assert moisture_excess(23.3, 66.0, 18.0, 86.0) > 0
    assert moisture_excess(23.3, 66.0, 19.5, 86.0) < 0


def test_surface_temperature_uses_iso13788_factor():
    # T_si = T_e + f * (T_i - T_e)
    assert surface_temperature(20.0, -5.0, 0.70) == pytest.approx(12.5, abs=0.01)
    assert surface_temperature(23.3, 7.3, 1.0) == pytest.approx(23.3, abs=0.01)
    assert surface_temperature(23.3, 7.3, 0.0) == pytest.approx(7.3, abs=0.01)


def test_surface_temperature_defaults_to_din_minimum():
    assert surface_temperature(20.0, -5.0) == pytest.approx(
        surface_temperature(20.0, -5.0, 0.70), abs=1e-9
    )


def test_din_4108_2_boundary_case():
    """The published acceptance case for the mould check.

    20 degC / 50 % indoors, -5 degC outdoors, R_si = 0.25 m2K/W: the surface must
    stay at or above 12.6 degC, which is f_Rsi = 0.70.
    """
    required_surface = 12.6
    # At that surface temperature the surface RH must be at the 80 % criterion.
    surface_rh = surface_relative_humidity(
        20.0, 50.0, -5.0, (required_surface - (-5.0)) / (20.0 - (-5.0))
    )
    assert surface_rh == pytest.approx(CRITICAL_SURFACE_RH, abs=0.5)

    # And solving the other way round reproduces the normative factor.
    e_indoor = vapour_pressure(20.0, 50.0)
    e_s_needed = e_indoor / (CRITICAL_SURFACE_RH / 100.0)
    # Invert Magnus for the required surface temperature.
    ratio = math.log(e_s_needed / 6.112)
    t_si = 243.12 * ratio / (17.62 - ratio)
    assert t_si == pytest.approx(12.62, abs=0.05)
    f_rsi = (t_si - (-5.0)) / (20.0 - (-5.0))
    assert f_rsi == pytest.approx(0.705, abs=0.005)


def test_surface_rh_exceeds_air_rh_on_a_cold_bridge():
    """Comfortable room air can still sit on a mouldy surface."""
    air_rh = 60.0
    surface_rh = surface_relative_humidity(23.3, air_rh, 7.3, 0.70)
    assert surface_rh > air_rh
    assert surface_rh > CRITICAL_SURFACE_RH


def test_surface_rh_capped_at_saturation():
    assert surface_relative_humidity(23.5, 70.0, -15.0, 0.50) == pytest.approx(
        100.0, abs=1e-9
    )


def test_max_safe_indoor_rh_slides_with_outdoor_temperature():
    """The ceiling must fall as it gets colder outside; a fixed 60 % cannot work."""
    mild = max_safe_indoor_rh(23.3, 15.0, 0.70)
    autumn = max_safe_indoor_rh(23.3, 7.3, 0.70)
    winter = max_safe_indoor_rh(23.3, -5.0, 0.70)
    assert mild > autumn > winter
    # Published working figure for the winter case.
    assert winter == pytest.approx(47.0, abs=2.0)
    # In autumn a fixed 60 % threshold is already too permissive here.
    assert autumn < 60.0


def test_max_safe_indoor_rh_is_consistent_with_surface_rh():
    """Sitting exactly at the ceiling puts the surface exactly at the criterion."""
    ceiling = max_safe_indoor_rh(23.3, 7.3, 0.70)
    surface_rh = surface_relative_humidity(23.3, ceiling, 7.3, 0.70)
    assert surface_rh == pytest.approx(CRITICAL_SURFACE_RH, abs=0.1)


def test_better_junction_allows_higher_indoor_humidity():
    poor = max_safe_indoor_rh(23.3, 7.3, 0.60)
    good = max_safe_indoor_rh(23.3, 7.3, 0.90)
    assert good > poor


def test_helpers_tolerate_missing_and_garbage_input():
    for bad in (None, "", "n/a", float("nan"), float("inf")):
        assert saturation_pressure(bad) is None
        assert absolute_humidity(bad, 50.0) is None
        assert absolute_humidity(20.0, bad) is None
        assert dew_point(bad, 50.0) is None
        assert moisture_excess(20.0, 50.0, bad, 80.0) is None
        assert surface_relative_humidity(bad, 50.0, 0.0, 0.7) is None
        assert max_safe_indoor_rh(bad, 0.0, 0.7) is None


def test_relative_humidity_is_clamped_to_physical_range():
    assert vapour_pressure(20.0, 150.0) == pytest.approx(
        saturation_pressure(20.0), abs=1e-9
    )
    assert vapour_pressure(20.0, -20.0) == pytest.approx(0.0, abs=1e-9)
