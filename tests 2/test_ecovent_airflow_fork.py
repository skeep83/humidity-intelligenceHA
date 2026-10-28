"""Fork tests: one-way airflow for EcoVent/Blauberg units (ecovent_v2).

These units have no siku ``device_type``, so zone airflow could not be applied
to them at all. Their airflow is a select on the same device. A zone with
``airflow_mode`` exhaust/supply now drives that select, but only:

- in the daytime zone lane (never during night soft extraction), and
- while the zone's air holds clearly more water than the outdoor air.

Every other case hands the unit back to ``heat_recovery``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any, Dict, List

from hi_runtime_fixtures import _FakeState
from test_dynamic_zones_fork import _build, _calls_to, _config, _load_airflow_module, _states, _zone

FAN = "fan.hi_fixture_zone3"
SELECT = "select.hi_fixture_zone3_hi_fixture_zone3_airflow"
TWINFRESH_OPTIONS = ["ventilation", "heat_recovery", "air_supply"]
BREEZY_OPTIONS = ["ventilation", "heat_recovery", "air_supply", "extract"]
OUT_TEMP = "sensor.hi_fixture_outdoor_temperature"
OUT_RH = "sensor.hi_fixture_outdoor_humidity"
DRY_OUTSIDE = (10.0, 80.0)   # ~7.5 g/m3 vs ~11 g/m3 in a 21 degC / 61 % room
WET_OUTSIDE = (20.0, 95.0)   # ~16 g/m3: one-way flow would import water


def _fan(percentage: int = 33) -> _FakeState:
    return _FakeState(
        "on",
        {
            "percentage": percentage,
            "preset_mode": "manual",
            "preset_modes": ["off", "low", "medium", "high", "manual"],
            "direction": "forward",
        },
    )


def _select(current: str = "heat_recovery", options: List[str] = TWINFRESH_OPTIONS) -> _FakeState:
    return _FakeState(current, {"options": list(options)})


def _airflow_zone(**kw) -> Dict[str, Any]:
    return _zone(**{"outputs": [FAN], "output_level": 33, "airflow_mode": "exhaust", **kw})


def _outdoor_states(outdoor) -> Dict[str, Any]:
    if outdoor is None:
        return {}
    temp, rh = outdoor
    return {
        OUT_TEMP: _FakeState(str(temp), {"unit_of_measurement": "°C"}),
        OUT_RH: _FakeState(str(rh), {"unit_of_measurement": "%"}),
    }


def _engine(zone, *, select_state=None, outdoor=DRY_OUTSIDE, humidity=61, extra=None, **top):
    config = _config(
        {"zone3": zone},
        outdoor_temperature_entity=OUT_TEMP,
        outdoor_humidity_entity=OUT_RH,
        **top,
    )
    states = {FAN: _fan(), SELECT: select_state or _select()}
    states.update(_outdoor_states(outdoor))
    states.update(extra or {})
    engine_mod, hass, entry = _build(config, _states(states, humidity))
    return engine_mod.HIAutomationEngine(hass, entry), hass


def _select_calls(hass):
    return [c for c in hass.services.calls if c[0] == "select" and c[1] == "select_option"]


def _age_drying_window(engine, minutes: int = 11):
    state = engine._absolute_humidity_state["airflow_dry::zone3"]
    state["on_since"] = state["on_since"] - _dt.timedelta(minutes=minutes)


def _run(engine, *, aged: bool = True):
    asyncio.run(engine._evaluate())
    if aged and "airflow_dry::zone3" in engine._absolute_humidity_state:
        state = engine._absolute_humidity_state["airflow_dry::zone3"]
        if state.get("on_since") is not None:
            _age_drying_window(engine)
            asyncio.run(engine._evaluate())


def _stop(engine):
    asyncio.run(engine.async_stop())


def test_exhaust_zone_selects_extraction_once_outdoor_air_is_drier():
    engine, hass = _engine(_airflow_zone())
    try:
        _run(engine)
        assert _select_calls(hass) == [
            ("select", "select_option", {"entity_id": SELECT, "option": "ventilation"}, False)
        ], hass.services.calls
        # Siku direction is not used for these units.
        assert _calls_to(hass.services.calls, FAN, "set_direction") == []
    finally:
        _stop(engine)


def test_drying_window_is_held_before_switching():
    engine, hass = _engine(_airflow_zone())
    try:
        _run(engine, aged=False)
        assert _select_calls(hass) == [], "one-way flow must wait for the 10 min hold"
    finally:
        _stop(engine)


def test_breezy_uses_its_extract_option():
    engine, hass = _engine(_airflow_zone(), select_state=_select(options=BREEZY_OPTIONS))
    try:
        _run(engine)
        assert [c[2]["option"] for c in _select_calls(hass)] == ["extract"]
    finally:
        _stop(engine)


def test_supply_zone_selects_air_supply():
    engine, hass = _engine(_zone(outputs=[FAN], output_level=33, airflow_mode="supply"))
    try:
        _run(engine)
        assert [c[2]["option"] for c in _select_calls(hass)] == ["air_supply"]
    finally:
        _stop(engine)


def test_wetter_outdoor_air_keeps_or_restores_heat_recovery():
    engine, hass = _engine(
        _airflow_zone(), select_state=_select("ventilation"), outdoor=WET_OUTSIDE
    )
    try:
        _run(engine)
        assert [c[2]["option"] for c in _select_calls(hass)] == ["heat_recovery"]
    finally:
        _stop(engine)


def test_no_outdoor_reference_means_no_one_way_flow():
    engine, hass = _engine(_airflow_zone(), outdoor=None)
    try:
        _run(engine)
        assert _select_calls(hass) == []
    finally:
        _stop(engine)


def test_idle_zone_hands_unit_back_to_heat_recovery():
    engine, hass = _engine(_airflow_zone(), select_state=_select("ventilation"), humidity=40)
    try:
        _run(engine)
        assert [c[2]["option"] for c in _select_calls(hass)] == ["heat_recovery"]
    finally:
        _stop(engine)


def test_heat_recovery_is_not_resent():
    engine, hass = _engine(_airflow_zone(), humidity=40)
    try:
        _run(engine)
        assert _select_calls(hass) == []
    finally:
        _stop(engine)


def test_night_soft_extraction_stays_in_heat_recovery():
    now_local = _dt.datetime.now().astimezone()
    start = (now_local.hour + 6) % 24
    end = (now_local.hour + 12) % 24
    engine, hass = _engine(
        _airflow_zone(night_bypass=True, output_level=66),
        select_state=_select("ventilation"),
        time_gate={
            "enabled": True,
            "start": f"{start:02d}:00",
            "end": f"{end:02d}:00",
            "outside_action": "safe_state",
        },
    )
    try:
        _run(engine)
        assert [c[2]["option"] for c in _select_calls(hass)] == ["heat_recovery"]
        speeds = [c[2]["percentage"] for c in hass.services.calls if c[1] == "set_percentage"]
        assert speeds == [], "fan already at the 33 % night cap"
    finally:
        _stop(engine)


def test_zone_without_airflow_mode_never_touches_the_select():
    engine, hass = _engine(
        _zone(outputs=[FAN], output_level=33, airflow_mode="none"),
        select_state=_select("ventilation"),
    )
    try:
        _run(engine)
        assert _select_calls(hass) == []
    finally:
        _stop(engine)


def test_manual_override_sends_no_airflow_change():
    engine, hass = _engine(_airflow_zone(), select_state=_select("ventilation"), humidity=40)
    hass.data["humidity_intelligence"][engine.entry.entry_id]["hi_input_booleans"][
        "air_control_manual_override"
    ].is_on = True
    try:
        _run(engine)
        assert _select_calls(hass) == []
    finally:
        _stop(engine)


def test_siku_units_keep_direction_control():
    engine, hass = _engine(
        _airflow_zone(),
        # OPUS (device_type 17): exhaust = reverse, the fan reports forward.
        extra={"sensor.hi_fixture_zone3_device_type": _FakeState("17", {})},
    )
    try:
        _run(engine)
        assert _select_calls(hass) == []
        assert _calls_to(hass.services.calls, FAN, "set_direction") == [
            ("fan", "set_direction", {"entity_id": FAN, "direction": "reverse"}, False)
        ], hass.services.calls
    finally:
        _stop(engine)


def test_airflow_select_option_mapping():
    airflow_select_option = _load_airflow_module().airflow_select_option

    assert airflow_select_option("exhaust", TWINFRESH_OPTIONS) == "ventilation"
    assert airflow_select_option("exhaust", BREEZY_OPTIONS) == "extract"
    assert airflow_select_option("supply", TWINFRESH_OPTIONS) == "air_supply"
    assert airflow_select_option("none", TWINFRESH_OPTIONS) == "heat_recovery"
    assert airflow_select_option("exhaust", ["heat_recovery"]) is None
