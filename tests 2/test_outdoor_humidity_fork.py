"""Fork tests: outdoor-aware humidity control.

These cover the two triggers that fix the physics of the original logic:

``moisture_excess``
    Ventilation removes water only while indoor absolute humidity exceeds the
    outdoor one. A fixed indoor-RH threshold cannot see that and fails in both
    directions - it ventilates on a warm humid evening (importing water) and
    stays idle when cheap drying is available.

``surface_mould_risk``
    Mould germinates above 80 % RH at the *surface* (ISO 13788 / BS 5250), which
    a comfortable-looking room reaches on a cold thermal bridge. Because the
    surface follows the outdoor temperature through f_Rsi, the effective ceiling
    slides with the weather instead of being a constant.

Measured site data used below (Bubuieci, 16.10): outdoor 7.3 degC / 86 %,
house 23.3 degC / 66 % -> excess ~ +7.0 g/m3.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any, Dict

import hi_runtime_fixtures as fixtures
from hi_runtime_fixtures import (
    BASELINE_TELEMETRY,
    _FakeHass,
    _FakeState,
    _load_runtime_modules,
)

OUT_TEMP = "sensor.hi_fixture_outdoor_temperature"
OUT_RH = "sensor.hi_fixture_outdoor_humidity"

# Note for future edits: _zone_trigger_level returns the fan level as a STRING
# ("66", "100"), not an int - comparing against a bare number fails even when the
# trigger logic is correct.


def _zone(**overrides) -> Dict[str, Any]:
    zone = {
        "enabled": True,
        "level": "level1",
        "rooms": ["Kitchen"],
        "outputs": [],
        "triggers": ["moisture_excess"],
        "thresholds": {"moisture_excess": 1.5},
        "output_level": 66,
        "boost_output_level": 100,
        "priority": 10,
        "airflow_mode": "none",
        "airflow_supply_direction": "auto",
        "night_bypass": False,
        "moisture_excess_on_hold_minutes": 0,
        "moisture_excess_off_hold_minutes": 0,
        "ui_label": "Outdoor-aware zone",
    }
    zone.update(overrides)
    return zone


def _config(zones: Dict[str, Any], **top) -> Dict[str, Any]:
    config = copy.deepcopy(fixtures._base_entry_data())
    config["zones"] = zones
    config["alert_handling_enabled"] = False
    config["aq"] = {}
    config.update(top)
    return config


def _states(indoor_t: float, indoor_rh: float, outdoor_t=None, outdoor_rh=None):
    states = {
        entity_id: _FakeState(value, fixtures._default_attrs(entity_id))
        for entity_id, value in BASELINE_TELEMETRY.items()
    }
    states["sensor.hi_fixture_kitchen_humidity"] = _FakeState(
        str(indoor_rh), {"unit_of_measurement": "%"}
    )
    states["sensor.hi_fixture_kitchen_temperature"] = _FakeState(
        str(indoor_t), {"unit_of_measurement": "\u00b0C"}
    )
    if outdoor_t is not None:
        states[OUT_TEMP] = _FakeState(str(outdoor_t), {"unit_of_measurement": "\u00b0C"})
    if outdoor_rh is not None:
        states[OUT_RH] = _FakeState(str(outdoor_rh), {"unit_of_measurement": "%"})
    return states


def _engine(config: Dict[str, Any], states: Dict[str, Any]):
    engine_mod, _core = _load_runtime_modules()
    entry = SimpleNamespace(entry_id=fixtures.ENTRY_ID, data=config, options={})
    hass = _FakeHass(entry, states)
    hass.data["humidity_intelligence"][fixtures.ENTRY_ID]["hi_input_booleans"][
        "air_isolate_fan_outputs"
    ].is_on = False
    return engine_mod.HIAutomationEngine(hass, entry)


def _evaluate_zone(zone: Dict[str, Any], states: Dict[str, Any], **top):
    """Return (selected_level, details) for a single zone."""
    config = _config({"zone1": zone}, **top)
    engine = _engine(config, states)
    try:
        level, details, _facts = engine._zone_trigger_level(
            zone.get("triggers", []), zone, zone.get("level"), zone_key="zone1"
        )
    finally:
        asyncio.run(engine.async_stop())
    return level, details


def _outdoor_config(**extra):
    return {
        "outdoor_temperature_entity": OUT_TEMP,
        "outdoor_humidity_entity": OUT_RH,
        **extra,
    }


# --------------------------------------------------------------------------
# moisture_excess
# --------------------------------------------------------------------------


def test_fires_when_outdoor_air_is_drier():
    """Real site conditions: +7.3 degC / 86 % outside dries a 23.3 / 66 % house."""
    level, details = _evaluate_zone(
        _zone(),
        _states(23.3, 66, outdoor_t=7.3, outdoor_rh=86),
        **_outdoor_config(),
    )
    assert level == "66"
    assert any("Moisture excess" in d for d in details)
    # ~7 g/m3 of headroom should be reported, not the raw RH.
    assert any("7.0 g/m3" in d or "6.9 g/m3" in d for d in details)


def test_silent_when_outdoor_air_is_wetter():
    """Warm humid evening: the old RH>60 rule would ventilate and import water."""
    level, details = _evaluate_zone(
        _zone(),
        _states(24.0, 62, outdoor_t=18.0, outdoor_rh=95),
        **_outdoor_config(),
    )
    assert level is None
    assert details == []


def test_silent_without_outdoor_sensors():
    """Backwards compatible: zones configured before this feature keep working."""
    level, details = _evaluate_zone(_zone(), _states(23.3, 66))
    assert level is None
    assert details == []


def test_silent_when_outdoor_sensor_unavailable():
    states = _states(23.3, 66, outdoor_t=7.3, outdoor_rh=86)
    states[OUT_RH] = _FakeState("unavailable", {})
    level, _details = _evaluate_zone(_zone(), states, **_outdoor_config())
    assert level is None


def test_threshold_is_respected():
    """A thin margin must not trigger a 5 g/m3 threshold."""
    zone = _zone(thresholds={"moisture_excess": 5.0})
    level, _details = _evaluate_zone(
        zone,
        _states(22.0, 55, outdoor_t=15.0, outdoor_rh=80),
        **_outdoor_config(),
    )
    assert level is None


def test_other_triggers_still_work_alongside():
    """Adding the new trigger must not disturb the existing absolute one."""
    zone = _zone(
        triggers=["humidity_absolute_high", "moisture_excess"],
        thresholds={"humidity_absolute_high": 60, "moisture_excess": 1.5},
        humidity_on_hold_minutes=0,
        humidity_off_hold_minutes=0,
        humidity_hysteresis=0,
    )
    level, details = _evaluate_zone(
        zone, _states(23.3, 66, outdoor_t=7.3, outdoor_rh=86), **_outdoor_config()
    )
    assert level == "66"
    assert any("absolute threshold" in d for d in details)
    assert any("Moisture excess" in d for d in details)


# --------------------------------------------------------------------------
# surface_mould_risk
# --------------------------------------------------------------------------


def _mould_zone(**overrides):
    zone = _zone(
        triggers=["surface_mould_risk"],
        thresholds={"surface_mould_risk": 80},
    )
    zone.update(overrides)
    return zone


def test_surface_risk_fires_though_room_air_looks_comfortable():
    """60 % in the room is already above the criterion on an f_Rsi 0.70 bridge."""
    level, details = _evaluate_zone(
        _mould_zone(f_rsi=0.70),
        _states(23.3, 60, outdoor_t=7.3, outdoor_rh=86),
        **_outdoor_config(),
    )
    # Risk triggers escalate to the boost level.
    assert level == "100"
    assert any("Surface humidity" in d for d in details)


def test_good_junction_does_not_fire_at_the_same_humidity():
    """Same air, a well-insulated corner: no mould risk."""
    level, _details = _evaluate_zone(
        _mould_zone(f_rsi=0.95),
        _states(23.3, 60, outdoor_t=7.3, outdoor_rh=86),
        **_outdoor_config(),
    )
    assert level is None


def test_ceiling_tightens_as_it_gets_colder():
    """50 % is safe in autumn on this junction but not in a hard frost."""
    autumn, _ = _evaluate_zone(
        _mould_zone(f_rsi=0.70),
        _states(23.3, 50, outdoor_t=7.3, outdoor_rh=86),
        **_outdoor_config(),
    )
    winter, _ = _evaluate_zone(
        _mould_zone(f_rsi=0.70),
        _states(23.3, 50, outdoor_t=-15.0, outdoor_rh=90),
        **_outdoor_config(),
    )
    assert autumn is None
    assert winter == "100"


def test_surface_risk_needs_outdoor_temperature():
    level, _details = _evaluate_zone(_mould_zone(), _states(23.3, 70))
    assert level is None


def test_f_rsi_is_clamped_to_sane_bounds():
    engine = _engine(
        _config({"zone1": _zone()}, **_outdoor_config()),
        _states(23.3, 66, outdoor_t=7.3, outdoor_rh=86),
    )
    try:
        assert engine._zone_f_rsi({"f_rsi": 5.0}) == 1.0
        assert engine._zone_f_rsi({"f_rsi": -3.0}) == 0.4
        assert engine._zone_f_rsi({"f_rsi": "nonsense"}) == 0.70
        assert engine._zone_f_rsi({}) == 0.70
    finally:
        asyncio.run(engine.async_stop())


def test_outdoor_rows_never_enter_house_averages():
    """The outdoor sensor must not be averaged into the house figures.

    This is why outdoor lives in its own config key instead of the telemetry
    list: an outdoor row would drag every average and every derived threshold.
    """
    config = _config({"zone1": _zone()}, **_outdoor_config())
    engine = _engine(config, _states(23.3, 66, outdoor_t=7.3, outdoor_rh=86))
    try:
        house_rh = engine._level_avg("humidity", None)
        house_temp = engine._level_avg("temperature", None)
    finally:
        asyncio.run(engine.async_stop())
    # Outdoor 7.3 degC / 86 % would visibly drag both if it were counted.
    assert house_temp is not None and house_temp > 15.0
    assert house_rh is not None and house_rh < 70.0
