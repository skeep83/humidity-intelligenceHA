"""Fork-specific tests: dynamic zones, airflow direction, absolute humidity,
night bypass, and the tolerant fan-percentage dedup.

These validate behavior added by this fork on top of upstream HI (which only
supported zone1/zone2 with direction-less fan control).
"""

from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

import hi_runtime_fixtures as fixtures
from hi_runtime_fixtures import (
    BASELINE_TELEMETRY,
    _FakeHass,
    _FakeState,
    _load_runtime_modules,
    run_air_control_simulation,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTEGRATION_ROOT = ROOT / "custom_components" / "humidity_intelligence"

FAN_OFF = _FakeState("off", {"percentage": 0})
HIGH_HUMIDITY = 61  # above the 60% absolute threshold used by _zone()


def _zone(**overrides) -> Dict[str, Any]:
    zone = {
        "enabled": True,
        "level": "level1",
        "rooms": ["Kitchen"],
        "outputs": [],
        "triggers": ["humidity_absolute_high"],
        "thresholds": {"humidity_absolute_high": 60},
        "output_level": 66,
        "boost_output_level": 100,
        "priority": 10,
        "airflow_mode": "none",
        "airflow_supply_direction": "auto",
        "night_bypass": False,
        "humidity_on_hold_minutes": 0,
        "humidity_off_hold_minutes": 0,
        "humidity_hysteresis": 0,
        "ui_label": "Test zone",
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


def _states(extra: Dict[str, Any], humidity: float = HIGH_HUMIDITY) -> Dict[str, Any]:
    states = {
        entity_id: _FakeState(value, fixtures._default_attrs(entity_id))
        for entity_id, value in BASELINE_TELEMETRY.items()
    }
    states["sensor.hi_fixture_kitchen_humidity"] = _FakeState(
        str(humidity), {"unit_of_measurement": "%"}
    )
    states.update(extra)
    return states


def _build(config: Dict[str, Any], states: Dict[str, Any]):
    engine_mod, _core_mod = _load_runtime_modules()
    entry = SimpleNamespace(entry_id=fixtures.ENTRY_ID, data=config, options={})
    hass = _FakeHass(entry, states)
    hass.data["humidity_intelligence"][fixtures.ENTRY_ID]["hi_input_booleans"][
        "air_isolate_fan_outputs"
    ].is_on = False
    return engine_mod, hass, entry


def _run_engine(config: Dict[str, Any], extra_states: Dict[str, Any], humidity: float = HIGH_HUMIDITY):
    """Run one engine evaluation; returns (fan_service_calls, airflow_targets)."""
    engine_mod, hass, entry = _build(config, _states(extra_states, humidity))
    engine = engine_mod.HIAutomationEngine(hass, entry)
    try:
        asyncio.run(engine._evaluate())
        airflow_targets = dict(engine._airflow_targets)
    finally:
        asyncio.run(engine.async_stop())
    fan_calls = [call for call in hass.services.calls if call[0] in {"fan", "switch"}]
    return fan_calls, airflow_targets


def _calls_to(fan_calls, entity_id: str, service: str):
    return [
        call for call in fan_calls
        if call[1] == service and call[2].get("entity_id") == entity_id
    ]


# ---------------------------------------------------------------------------
# Dynamic zones
# ---------------------------------------------------------------------------

def test_third_zone_runs_concurrently_with_first_two():
    """Upstream HI let zone1 suppress zone2; this fork evaluates every zone
    and arbitrates per output, so three zones can act at the same time."""
    config = _config({
        "zone1": _zone(outputs=["fan.hi_fixture_zone1"], priority=10),
        "zone2": _zone(outputs=["fan.hi_fixture_zone2"], priority=20),
        "zone3": _zone(outputs=["fan.hi_fixture_zone3"], priority=30),
    })
    fan_calls, _engine = _run_engine(config, {
        "fan.hi_fixture_zone1": FAN_OFF,
        "fan.hi_fixture_zone2": FAN_OFF,
        "fan.hi_fixture_zone3": FAN_OFF,
    })
    targeted = {
        call[2].get("entity_id") for call in fan_calls if call[1] == "set_percentage"
    }
    assert targeted == {
        "fan.hi_fixture_zone1",
        "fan.hi_fixture_zone2",
        "fan.hi_fixture_zone3",
    }, fan_calls


def test_zone_priority_decides_the_winner_on_a_shared_output():
    """When two zones claim the same fan, the lower priority number wins."""
    config = _config({
        "zone1": _zone(
            outputs=["fan.hi_fixture_shared"], priority=30,
            output_level=100, ui_label="Low prio",
        ),
        "zone3": _zone(
            outputs=["fan.hi_fixture_shared"], priority=10,
            output_level=33, ui_label="High prio",
        ),
    })
    fan_calls, _engine = _run_engine(config, {"fan.hi_fixture_shared": FAN_OFF})
    percentages = [
        call[2].get("percentage")
        for call in _calls_to(fan_calls, "fan.hi_fixture_shared", "set_percentage")
    ]
    assert percentages == [33], fan_calls


def test_disabled_zone_does_not_drive_or_reset_its_outputs():
    """A disabled zone must neither drive nor reset its outputs (upstream
    reset disabled-zone outputs to auto and fought external automations)."""
    config = _config({
        "zone1": _zone(outputs=["fan.hi_fixture_zone1"], priority=10),
        "zone2": _zone(outputs=["fan.hi_fixture_zone2"], priority=20, enabled=False),
    })
    fan_calls, _engine = _run_engine(config, {
        "fan.hi_fixture_zone1": FAN_OFF,
        # zone2 is disabled: this fan must not receive any command.
        "fan.hi_fixture_zone2": _FakeState("on", {"percentage": 33, "preset_mode": "manual"}),
    })
    zone2_calls = [
        call for call in fan_calls if call[2].get("entity_id") == "fan.hi_fixture_zone2"
    ]
    assert zone2_calls == [], zone2_calls
    assert _calls_to(fan_calls, "fan.hi_fixture_zone1", "set_percentage"), fan_calls


# ---------------------------------------------------------------------------
# Airflow direction
# ---------------------------------------------------------------------------

_VENTO_FAN = "fan.hi_fixture_siku_192_168_4_103"
_VENTO_DEVICE_TYPE = "sensor.siku_192_168_4_103_device_type"


def _airflow_config(mode: str, **zone_kw) -> Dict[str, Any]:
    return _config({
        "zone3": _zone(
            outputs=[_VENTO_FAN],
            airflow_mode=mode,
            ui_label="Master",
            **zone_kw,
        ),
    })


def _vento_states(fan_state=None):
    return {
        _VENTO_FAN: fan_state if fan_state is not None else FAN_OFF,
        _VENTO_DEVICE_TYPE: _FakeState("26"),
    }


def test_vento_exhaust_zone_sends_forward_direction():
    """VENTO (device_type 26): forward = exhaust."""
    fan_calls, _engine = _run_engine(_airflow_config("exhaust"), _vento_states())
    assert _calls_to(fan_calls, _VENTO_FAN, "set_direction") == [
        ("fan", "set_direction", {"entity_id": _VENTO_FAN, "direction": "forward"}, False)
    ], fan_calls


def test_vento_supply_zone_sends_reverse_direction():
    """VENTO (device_type 26): reverse = supply."""
    fan_calls, _engine = _run_engine(_airflow_config("supply"), _vento_states())
    assert _calls_to(fan_calls, _VENTO_FAN, "set_direction") == [
        ("fan", "set_direction", {"entity_id": _VENTO_FAN, "direction": "reverse"}, False)
    ], fan_calls


def test_direction_is_not_resent_when_already_correct():
    """No duplicate set_direction while the fan already reports the target,
    but the engine still tracks it as an active airflow target."""
    fan_calls, airflow_targets = _run_engine(
        _airflow_config("exhaust"),
        _vento_states(_FakeState("on", {"percentage": 66, "direction": "forward"})),
    )
    assert _calls_to(fan_calls, _VENTO_FAN, "set_direction") == []
    assert airflow_targets == {_VENTO_FAN: "forward"}


def test_direction_suppressed_when_device_type_unknown():
    """Without device_type and without an override the fork must not guess,
    but speed control still works."""
    fan_calls, airflow_targets = _run_engine(
        _airflow_config("exhaust"),
        {_VENTO_FAN: FAN_OFF},
    )
    assert _calls_to(fan_calls, _VENTO_FAN, "set_direction") == []
    assert _calls_to(fan_calls, _VENTO_FAN, "set_percentage"), fan_calls
    assert airflow_targets == {}


def test_zone_supply_direction_override_wins_over_detection():
    """airflow_supply_direction=forward forces supply=forward even on VENTO."""
    fan_calls, _engine = _run_engine(
        _airflow_config("supply", airflow_supply_direction="forward"),
        _vento_states(),
    )
    assert _calls_to(fan_calls, _VENTO_FAN, "set_direction") == [
        ("fan", "set_direction", {"entity_id": _VENTO_FAN, "direction": "forward"}, False)
    ], fan_calls


def test_airflow_helpers_resolve_opus_and_vento_semantics():
    """Unit-level mapping table check for both recuperator families."""
    airflow = _load_airflow_module()
    from hi_runtime_fixtures import _FakeStates

    hass_opus = SimpleNamespace(
        states=_FakeStates({"sensor.opus_1_device_type": _FakeState("17")})
    )
    hass_vento = SimpleNamespace(
        states=_FakeStates({"sensor.vento_1_device_type": _FakeState("26")})
    )
    plan = airflow.resolve_output_airflow(hass_opus, "fan.opus_1", "supply", {})
    assert plan.siku_direction == "reverse" and plan.source == "auto"
    plan = airflow.resolve_output_airflow(hass_opus, "fan.opus_1", "exhaust", {})
    assert plan.siku_direction == "forward"
    plan = airflow.resolve_output_airflow(hass_vento, "fan.vento_1", "supply", {})
    assert plan.siku_direction == "reverse" and plan.source == "auto"
    plan = airflow.resolve_output_airflow(hass_vento, "fan.vento_1", "exhaust", {})
    assert plan.siku_direction == "forward"
    plan = airflow.resolve_output_airflow(
        hass_opus, "fan.x", "supply", {"fan.x": {"supply_direction": "reverse"}}
    )
    assert plan.siku_direction == "reverse" and plan.source == "override"
    plan = airflow.resolve_output_airflow(
        SimpleNamespace(states=None), "fan.unknown_1", "supply", {}
    )
    assert plan.siku_direction is None and plan.source == "unknown"


def _load_airflow_module():
    pkg = "hi_airflow_testpkg"
    if pkg not in sys.modules:
        root = types.ModuleType(pkg)
        root.__path__ = [str(ROOT)]
        sys.modules[pkg] = root
        helpers = types.ModuleType(f"{pkg}.helpers")
        helpers.__path__ = [str(INTEGRATION_ROOT / "helpers")]
        sys.modules[f"{pkg}.helpers"] = helpers
    spec = importlib.util.spec_from_file_location(
        f"{pkg}.helpers.airflow", INTEGRATION_ROOT / "helpers" / "airflow.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg}.helpers.airflow"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Absolute humidity hysteresis
# ---------------------------------------------------------------------------

def _hold_zone(**kw) -> Dict[str, Any]:
    return _zone(
        outputs=["fan.hi_fixture_zone3"],
        humidity_on_hold_minutes=20,
        humidity_off_hold_minutes=45,
        humidity_hysteresis=5,
        **kw,
    )


def test_absolute_humidity_hold_blocks_instant_trigger():
    """With a 20-minute on-hold, a fresh spike must NOT trigger immediately."""
    engine_mod, hass, entry = _build(
        _config({"zone3": _hold_zone()}),
        _states({"fan.hi_fixture_zone3": FAN_OFF}),
    )
    engine = engine_mod.HIAutomationEngine(hass, entry)
    try:
        asyncio.run(engine._evaluate())
        assert _calls_to(
            [c for c in hass.services.calls if c[0] == "fan"],
            "fan.hi_fixture_zone3",
            "set_percentage",
        ) == []
        state = engine._absolute_humidity_state["zone3"]
        assert state["active"] is False and state["on_since"] is not None
    finally:
        asyncio.run(engine.async_stop())


def test_absolute_humidity_hold_releases_after_window():
    """After the hold window elapsed with sustained high humidity, the zone
    engages on the next evaluation."""
    config = _config({"zone3": _hold_zone()})
    engine_mod, hass, entry = _build(config, _states({"fan.hi_fixture_zone3": FAN_OFF}))
    engine = engine_mod.HIAutomationEngine(hass, entry)
    try:
        asyncio.run(engine._evaluate())
        assert [c for c in hass.services.calls if c[1] == "set_percentage"] == []
        # Pretend the spike started 21 minutes ago and re-evaluate.
        state = engine._absolute_humidity_state["zone3"]
        assert state["on_since"] is not None
        state["on_since"] = state["on_since"] - _dt.timedelta(minutes=21)
        asyncio.run(engine._evaluate())
        engaged = _calls_to(
            [c for c in hass.services.calls if c[0] == "fan"],
            "fan.hi_fixture_zone3",
            "set_percentage",
        )
        assert engaged, hass.services.calls
        assert engine._absolute_humidity_state["zone3"]["active"] is True
    finally:
        asyncio.run(engine.async_stop())


def test_absolute_hysteresis_band_freezes_release():
    """Inside the band (between threshold-hysteresis and threshold) an active
    trigger must stay active: no off-timer accumulation."""
    config = _config({
        "zone3": _zone(
            outputs=["fan.hi_fixture_zone3"],
            humidity_on_hold_minutes=0,
            humidity_off_hold_minutes=45,
            humidity_hysteresis=5,
        ),
    })
    engine_mod, hass, entry = _build(
        config, _states({"fan.hi_fixture_zone3": FAN_OFF}, humidity=62)
    )
    engine = engine_mod.HIAutomationEngine(hass, entry)
    try:
        asyncio.run(engine._evaluate())
        assert engine._absolute_humidity_state["zone3"]["active"] is True
        # Drop into the band (57% is between 55 and 60): state must freeze.
        hass.states._values["sensor.hi_fixture_kitchen_humidity"].state = "57"
        asyncio.run(engine._evaluate())
        state = engine._absolute_humidity_state["zone3"]
        assert state["active"] is True
        assert state["off_since"] is None
        # Drop below the band: off-timer starts but 45 min have not passed.
        hass.states._values["sensor.hi_fixture_kitchen_humidity"].state = "50"
        asyncio.run(engine._evaluate())
        state = engine._absolute_humidity_state["zone3"]
        assert state["active"] is True and state["off_since"] is not None
    finally:
        asyncio.run(engine.async_stop())


# ---------------------------------------------------------------------------
# Fan percentage dedup tolerance (32 vs 33)
# ---------------------------------------------------------------------------

def test_fan_percentage_dedup_tolerates_siku_rounding():
    """Target 33 with the fan reporting 32 must not re-send set_percentage."""
    config = _config({"zone3": _zone(outputs=["fan.hi_fixture_zone3"], output_level=33)})
    fan_calls, _engine = _run_engine(
        config,
        {"fan.hi_fixture_zone3": _FakeState("on", {"percentage": 32})},
    )
    assert _calls_to(fan_calls, "fan.hi_fixture_zone3", "set_percentage") == []
    assert _calls_to(fan_calls, "fan.hi_fixture_zone3", "turn_on") == []


def test_fan_percentage_outside_tolerance_is_still_sent():
    """A real mismatch (fan at 50, target 33) must still be corrected."""
    config = _config({"zone3": _zone(outputs=["fan.hi_fixture_zone3"], output_level=33)})
    fan_calls, _engine = _run_engine(
        config,
        {"fan.hi_fixture_zone3": _FakeState("on", {"percentage": 50})},
    )
    assert _calls_to(fan_calls, "fan.hi_fixture_zone3", "set_percentage") == [
        ("fan", "set_percentage", {"entity_id": "fan.hi_fixture_zone3", "percentage": 33}, False)
    ], fan_calls


# ---------------------------------------------------------------------------
# Night bypass through the time gate
# ---------------------------------------------------------------------------

def test_night_bypass_runs_capped_extraction_and_blocks_others():
    """Outside the time-gate window only night_bypass zones with an active
    absolute trigger may run, capped at the lowest fan step."""
    now_local = _dt.datetime.now().astimezone()
    start = ((now_local.hour + 6) % 24)
    end = ((now_local.hour + 12) % 24)
    config = _config(
        {
            "zone1": _zone(
                outputs=["fan.hi_fixture_zone1"],
                triggers=["humidity_absolute_high"],
                priority=10,
                output_level=100,
            ),
            "zone3": _zone(
                outputs=["fan.hi_fixture_zone3"],
                priority=30,
                night_bypass=True,
                airflow_mode="exhaust",
                humidity_on_hold_minutes=0,
                humidity_off_hold_minutes=45,
                humidity_hysteresis=5,
                output_level=66,
            ),
        },
        time_gate={
            "enabled": True,
            "start": f"{start:02d}:00",
            "end": f"{end:02d}:00",
            "outside_action": "safe_state",
        },
    )
    fan_calls, _engine = _run_engine(config, {
        "fan.hi_fixture_zone1": FAN_OFF,
        "fan.hi_fixture_zone3": FAN_OFF,
    })
    driving = {"set_percentage", "turn_on", "turn_off"}
    zone1 = [c for c in fan_calls if c[2].get("entity_id") == "fan.hi_fixture_zone1" and c[1] in driving]
    zone3 = [c for c in fan_calls if c[2].get("entity_id") == "fan.hi_fixture_zone3" and c[1] in driving]
    assert zone1 == [], f"non-bypass zone must stay gated at night: {zone1}"
    percentages = [c[2]["percentage"] for c in zone3 if c[1] == "set_percentage"]
    assert percentages == [33], f"night bypass must cap at step 33: {fan_calls}"


# ---------------------------------------------------------------------------
# Options-flow end to end: add a third zone through the UI steps
# ---------------------------------------------------------------------------

def test_options_flow_adds_third_zone_with_fork_fields():
    """Full UI path: zones menu -> ➕ Add a new zone -> edit form -> save.

    Verifies the fork's dynamic zone machinery persists a well-formed zone3
    with airflow/hysteresis fields, without touching zone1/zone2.
    """
    import test_config_flow_sanity as cfs

    config_flow = cfs._load_config_flow_module()
    entry = SimpleNamespace(
        data={
            "zones": {
                "zone1": {
                    "enabled": True, "level": "level1", "rooms": ["Kitchen"],
                    "triggers": ["humidity_high"], "outputs": ["fan.kitchen"],
                    "thresholds": {"humidity_high": 6}, "output_level": 33,
                    "boost_output_level": 66, "ui_label": "First floor",
                },
                "zone2": {
                    "enabled": False, "level": "level2", "rooms": [],
                    "triggers": [], "outputs": [], "thresholds": {},
                    "output_level": 33, "boost_output_level": 66, "ui_label": "Zone 2",
                },
            }
        },
        options={},
    )
    flow = config_flow.HumidityIntelligenceOptionsFlow(entry)
    flow.hass = SimpleNamespace()

    # The menu offers the existing zones plus ➕ add.
    menu = asyncio.run(flow.async_step_options_zones())
    values = cfs._schema_select_values(menu, "action")
    assert "zone1" in values and "zone2" in values and "zone_add" in values

    # ➕ Add a new zone lands on the edit form for zone3.
    form = asyncio.run(flow.async_step_options_zones({"action": "zone_add"}))
    assert form["step_id"] == "options_zone_edit"
    assert flow._pending_zone_key == "zone3"

    saved = asyncio.run(
        flow.async_step_options_zone_edit(
            {
                "enabled": True,
                "level": "level2",
                "rooms": ["Master Bedroom"],
                "triggers": ["humidity_absolute_high"],
                "outputs": ["fan.master_recuperator"],
                "priority": 15,
                "airflow_mode": "exhaust",
                "airflow_supply_direction": "auto",
                "night_bypass": True,
                "show_advanced_options": {
                    "output_level": 33,
                    "boost_output_level": 66,
                    "ui_label": "Master",
                    "humidity_hysteresis": 5.0,
                    "humidity_on_hold_minutes": 20,
                    "humidity_off_hold_minutes": 45,
                },
            }
        )
    )
    assert saved["step_id"] == "options_zones"  # back to the zones menu after save

    zones = flow._options["zones"]
    assert set(zones) == {"zone1", "zone2", "zone3"}
    zone3 = zones["zone3"]
    assert zone3["enabled"] is True
    assert zone3["rooms"] == ["Master Bedroom"]
    assert zone3["triggers"] == ["humidity_absolute_high"]
    assert zone3["outputs"] == ["fan.master_recuperator"]
    assert zone3["priority"] == 15
    assert zone3["airflow_mode"] == "exhaust"
    assert zone3["night_bypass"] is True
    assert zone3["humidity_hysteresis"] == 5.0
    assert zone3["humidity_on_hold_minutes"] == 20
    assert zone3["humidity_off_hold_minutes"] == 45
    assert zone3["thresholds"]["humidity_absolute_high"] == 60
    # Existing zones untouched.
    assert zones["zone1"]["ui_label"] == "First floor"
    assert zones["zone2"]["enabled"] is False

    # ➖ Remove a zone deletes only the chosen zone.
    remove_form = asyncio.run(flow.async_step_options_zone_remove())
    assert remove_form["step_id"] == "options_zone_remove"
    asyncio.run(flow.async_step_options_zone_remove({"zone_key": "zone3"}))
    assert set(flow._options["zones"]) == {"zone1", "zone2"}


# ---------------------------------------------------------------------------
# Presentation: dynamic zone labels in the runtime copy
# ---------------------------------------------------------------------------
def test_third_zone_pressure_is_represented_in_runtime_copy():
    """The air_control_mode sensor must reflect a zone3-driven state with the
    zone's own display label."""
    result = run_air_control_simulation(
        telemetry_overrides={"sensor.hi_fixture_kitchen_humidity": 61},
        config_overrides={
            "alert_handling_enabled": False,
            "zones": {
                "zone3": _zone(
                    outputs=["fan.hi_fixture_zone3"],
                    ui_label="Master Bedroom",
                )
            },
        },
        state_overrides={"fan.hi_fixture_zone3": FAN_OFF},
        boolean_overrides={"air_isolate_fan_outputs": False},
    )
    assert result.mode_sensor_state not in {"normal", "telemetry_unavailable"}
    reason_text = (result.runtime_reason + " " + result.reason_sensor_state).lower()
    assert "master bedroom" in reason_text, (
        result.runtime_reason,
        result.reason_sensor_state,
    )
