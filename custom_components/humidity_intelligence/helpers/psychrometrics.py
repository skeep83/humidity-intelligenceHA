"""Psychrometric helpers for Humidity Intelligence (fork feature).

Everything ventilation control needs that relative humidity alone cannot answer:
how much water the air actually carries, whether outdoor air can dry the house,
and how damp the coldest surface of a room gets.

Why this module exists
----------------------
Relative humidity is a ratio, not an amount: the saturation pressure roughly
doubles per +10 K, so the same RH means very different amounts of water, and RH
moves when only the temperature moves.  Ventilation drying is governed by a mass
balance of water vapour::

    V * d(nu_in)/dt = G - Q * (nu_in - nu_out)

so air exchange dries a room only while ``nu_in > nu_out`` and ADDS moisture when
``nu_in < nu_out``.  RH does not appear in that equation at all, which is why a
fixed "RH above 60 %" threshold both fires when outdoor air is wetter (adding
water) and stays silent when cheap drying is available.

Formulas
--------
Magnus saturation pressure (Sonntag 1990 coefficients, as published in the WMO
Guide to Instruments and Methods of Observation, WMO-No. 8, humidity chapter);
valid over water for roughly -45..+60 degC::

    e_s(t) = 6.112 * exp(17.62 * t / (243.12 + t))          [hPa]
    e      = RH / 100 * e_s(t)                              [hPa]
    Td     = 243.12 * L / (17.62 - L),  L = ln(e / 6.112)   [degC]

Absolute humidity from the ideal gas law for water vapour
(R_v = 461.5 J/(kg*K), ASHRAE Handbook - Fundamentals, Psychrometrics)::

    nu [g/m3] = 2.16679 * e[Pa] / T[K]

Surface condensation and mould use the temperature factor of ISO 13788::

    T_si = T_e + f_Rsi * (T_i - T_e)

with the critical surface humidity criterion of 80 % RH (ISO 13788, BS 5250:2011).
DIN 4108-2 requires f_Rsi >= 0.70 for dwellings, BRE IP1/06 uses 0.75.

The implementation is verified against the DIN 4108-2 boundary case in the unit
tests: 20 degC / 50 % indoors, -5 degC outdoors must yield a required surface
temperature of 12.6 degC and f_Rsi = 0.70.
"""

from __future__ import annotations

import math
from typing import Optional

# Magnus / Sonntag coefficients over water.
MAGNUS_A = 6.112  # hPa
MAGNUS_B = 17.62
MAGNUS_C = 243.12  # degC

# 1000 / R_v with R_v = 461.5 J/(kg*K), giving g/m3 from Pa and K.
VAPOUR_DENSITY_FACTOR = 2.16679

# Critical surface relative humidity for mould growth (ISO 13788, BS 5250:2011).
CRITICAL_SURFACE_RH = 80.0

# Minimum temperature factor required for dwellings (DIN 4108-2 section 6.2).
F_RSI_DEFAULT = 0.70

__all__ = [
    "CRITICAL_SURFACE_RH",
    "F_RSI_DEFAULT",
    "saturation_pressure",
    "vapour_pressure",
    "absolute_humidity",
    "dew_point",
    "surface_temperature",
    "surface_relative_humidity",
    "moisture_excess",
    "max_safe_indoor_rh",
]


def _finite(value: Optional[float]) -> Optional[float]:
    """Return value as float when it is a usable finite number, else None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def saturation_pressure(temperature_c: Optional[float]) -> Optional[float]:
    """Saturation vapour pressure over water in hPa."""
    temp = _finite(temperature_c)
    if temp is None:
        return None
    # Guard the Magnus pole at t = -243.12 degC; nothing habitable goes near it.
    if temp <= -MAGNUS_C + 1.0:
        return None
    return MAGNUS_A * math.exp(MAGNUS_B * temp / (MAGNUS_C + temp))


def vapour_pressure(
    temperature_c: Optional[float], relative_humidity: Optional[float]
) -> Optional[float]:
    """Partial vapour pressure in hPa from temperature and RH in percent."""
    e_s = saturation_pressure(temperature_c)
    rh = _finite(relative_humidity)
    if e_s is None or rh is None:
        return None
    rh = max(0.0, min(100.0, rh))
    return rh / 100.0 * e_s


def absolute_humidity(
    temperature_c: Optional[float], relative_humidity: Optional[float]
) -> Optional[float]:
    """Absolute humidity (water vapour density) in g/m3."""
    e = vapour_pressure(temperature_c, relative_humidity)
    temp = _finite(temperature_c)
    if e is None or temp is None:
        return None
    kelvin = temp + 273.15
    if kelvin <= 0:
        return None
    return VAPOUR_DENSITY_FACTOR * (e * 100.0) / kelvin


def dew_point(
    temperature_c: Optional[float], relative_humidity: Optional[float]
) -> Optional[float]:
    """Dew point in degC (inverse Magnus)."""
    e = vapour_pressure(temperature_c, relative_humidity)
    if e is None or e <= 0:
        return None
    ratio = math.log(e / MAGNUS_A)
    denominator = MAGNUS_B - ratio
    if abs(denominator) < 1e-9:
        return None
    return MAGNUS_C * ratio / denominator


def surface_temperature(
    indoor_c: Optional[float], outdoor_c: Optional[float], f_rsi: Optional[float] = None
) -> Optional[float]:
    """Temperature of the coldest surface from the ISO 13788 temperature factor.

    ``f_rsi`` is a property of the junction (geometry plus insulation), not of the
    weather, so it can be measured once and reused for any outdoor temperature.
    """
    t_in = _finite(indoor_c)
    t_out = _finite(outdoor_c)
    factor = _finite(f_rsi)
    if factor is None:
        factor = F_RSI_DEFAULT
    if t_in is None or t_out is None:
        return None
    factor = max(0.0, min(1.0, factor))
    return t_out + factor * (t_in - t_out)


def surface_relative_humidity(
    indoor_c: Optional[float],
    indoor_rh: Optional[float],
    outdoor_c: Optional[float],
    f_rsi: Optional[float] = None,
) -> Optional[float]:
    """Relative humidity at the coldest surface, in percent.

    This is the quantity the mould criterion is written against: room air may sit
    at a comfortable 60 % while the window reveal behind the curtain is already
    above 80 %.
    """
    e = vapour_pressure(indoor_c, indoor_rh)
    t_si = surface_temperature(indoor_c, outdoor_c, f_rsi)
    e_s_surface = saturation_pressure(t_si)
    if e is None or e_s_surface is None or e_s_surface <= 0:
        return None
    return min(100.0, e / e_s_surface * 100.0)


def moisture_excess(
    indoor_c: Optional[float],
    indoor_rh: Optional[float],
    outdoor_c: Optional[float],
    outdoor_rh: Optional[float],
) -> Optional[float]:
    """Water the room carries above outdoor air, in g/m3.

    Positive means ventilation removes water, negative means ventilation would
    import it.  This is the only correct switching variable for humidity-driven
    ventilation; comparing indoor RH against a constant is not.
    """
    nu_in = absolute_humidity(indoor_c, indoor_rh)
    nu_out = absolute_humidity(outdoor_c, outdoor_rh)
    if nu_in is None or nu_out is None:
        return None
    return nu_in - nu_out


def max_safe_indoor_rh(
    indoor_c: Optional[float],
    outdoor_c: Optional[float],
    f_rsi: Optional[float] = None,
    critical_surface_rh: float = CRITICAL_SURFACE_RH,
) -> Optional[float]:
    """Highest indoor RH that keeps the coldest surface below the mould criterion.

    This is the sliding ceiling that replaces a fixed "60 %" threshold: the colder
    it gets outside, the colder the thermal bridge, and the lower the indoor
    humidity has to stay.  With f_Rsi = 0.70 and -5 degC outdoors the ceiling
    falls to roughly 47 %, far below the usual fixed setting.
    """
    t_in = _finite(indoor_c)
    t_si = surface_temperature(indoor_c, outdoor_c, f_rsi)
    e_s_indoor = saturation_pressure(t_in)
    e_s_surface = saturation_pressure(t_si)
    limit = _finite(critical_surface_rh)
    if e_s_indoor is None or e_s_surface is None or limit is None or e_s_indoor <= 0:
        return None
    ceiling = limit / 100.0 * e_s_surface / e_s_indoor * 100.0
    return max(0.0, min(100.0, ceiling))
