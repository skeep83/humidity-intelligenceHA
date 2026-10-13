"""Airflow direction helpers for zone outputs (supply/exhaust semantics).

The physical meaning of siku's ``forward``/``reverse`` differs per recuperator
model family:

- Blauberg Freshpoint (OPUS, siku device_type 17): ``forward`` = supply.
- Blauberg VENTO Expert (siku device_type 26): ``reverse`` = supply.

Zones are configured with *semantic* directions (``supply`` / ``exhaust``) and
each output entity resolves them to a concrete siku direction through a
per-entity mapping. The mapping is auto-detected from the siku ``device_type``
sensor exposed on the same HA device and can be overridden in zone config.

This module is HA-state-aware but import-light: it only reads
``hass.states``/``hass.helpers`` at call time so unit tests can pass a stub.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# Semantic airflow values used in zone configuration.
AIRFLOW_NONE = "none"
AIRFLOW_SUPPLY = "supply"
AIRFLOW_EXTRACT = "exhaust"

# Concrete siku direction values accepted by fan.set_direction.
SIKU_FORWARD = "forward"
SIKU_REVERSE = "reverse"
SIKU_ALTERNATING = "alternating"

# device_type -> concrete siku direction that means "supply" for that family.
SUPPLY_DIRECTION_BY_DEVICE_TYPE: Dict[int, str] = {
    17: SIKU_FORWARD,   # Freshpoint OPUS: forward blows outside air in.
    26: SIKU_REVERSE,   # VENTO Expert: reverse blows outside air in.
}

# device_type -> concrete siku direction that means "exhaust".
EXHAUST_DIRECTION_BY_DEVICE_TYPE: Dict[int, str] = {
    17: SIKU_REVERSE,
    26: SIKU_FORWARD,
}

DEVICE_TYPE_SENSOR_SUFFIX = "_device_type"
VALID_SEMANTIC_AIRFLOW = (AIRFLOW_NONE, AIRFLOW_SUPPLY, AIRFLOW_EXTRACT)


@dataclass
class OutputAirflowPlan:
    """Resolved airflow command for one output entity."""

    entity_id: str
    semantic: str
    siku_direction: Optional[str]
    source: str  # override | auto | unknown
    device_type: Optional[int] = None
    reason: str = field(default="")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "semantic": self.semantic,
            "direction": self.siku_direction,
            "source": self.source,
            "device_type": self.device_type,
            "reason": self.reason,
        }


def normalize_semantic_airflow(value: Any) -> Optional[str]:
    """Return a valid semantic airflow value or None when unusable."""
    text = str(value or "").strip().lower()
    if text in VALID_SEMANTIC_AIRFLOW:
        return text
    return None


def resolve_output_airflow(
    hass: Any,
    entity_id: str,
    semantic: str,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> OutputAirflowPlan:
    """Resolve one output's semantic airflow into a concrete siku direction.

    ``overrides`` maps entity_id -> zone ``airflow_overrides`` entry
    (``{"supply_direction": "forward"|"reverse"}``). Overrides win over
    auto-detection. When the device type cannot be detected the plan carries
    ``source="unknown"`` and no direction, so callers must not send one.
    """
    semantic = normalize_semantic_airflow(semantic) or AIRFLOW_NONE
    override = (overrides or {}).get(entity_id) or {}
    device_type = detect_device_type(hass, entity_id)

    override_direction = str(override.get("supply_direction") or "").strip().lower()
    if override_direction in (SIKU_FORWARD, SIKU_REVERSE):
        supply_direction = override_direction
        exhaust_direction = SIKU_REVERSE if supply_direction == SIKU_FORWARD else SIKU_FORWARD
        source = "override"
    elif device_type in SUPPLY_DIRECTION_BY_DEVICE_TYPE:
        supply_direction = SUPPLY_DIRECTION_BY_DEVICE_TYPE[device_type]
        exhaust_direction = EXHAUST_DIRECTION_BY_DEVICE_TYPE[device_type]
        source = "auto"
    else:
        return OutputAirflowPlan(
            entity_id=entity_id,
            semantic=semantic,
            siku_direction=None,
            source="unknown",
            device_type=device_type,
            reason=(
                "device_type unavailable and no airflow override configured; "
                "direction commands are suppressed for this output"
            ),
        )

    if semantic == AIRFLOW_SUPPLY:
        direction = supply_direction
    elif semantic == AIRFLOW_EXTRACT:
        direction = exhaust_direction
    else:
        direction = SIKU_ALTERNATING

    return OutputAirflowPlan(
        entity_id=entity_id,
        semantic=semantic,
        siku_direction=direction,
        source=source,
        device_type=device_type,
        reason="",
    )


def detect_device_type(hass: Any, fan_entity_id: str) -> Optional[int]:
    """Find the siku device_type int for a fan entity, or None.

    Strategy (no device registry import required): collect candidate
    ``sensor.*_device_type`` entity ids from the entity registry when
    available, otherwise scan all ``*_device_type`` sensors. Prefer a sensor
    that shares the fan's device via the registry; fall back to a sensor whose
    entity id embeds the fan's IP (siku names entities by IP).
    """
    candidates: List[str] = []
    same_device_only = False
    registry = getattr(hass, "states", None)
    if registry is None:
        return None

    er = None
    try:
        from homeassistant.helpers import entity_registry as er_module

        er = er_module.async_get(hass)
    except Exception:  # pragma: no cover - registry unavailable in stubs
        er = None

    fan_entry = None
    if er is not None:
        try:
            fan_entry = er.async_get(fan_entity_id)
        except Exception:  # pragma: no cover
            fan_entry = None

    if fan_entry is not None and getattr(fan_entry, "device_id", None):
        try:
            same_device_only = True
            for entry in er.entities.values():
                if entry.device_id != fan_entry.device_id:
                    continue
                eid = str(entry.entity_id)
                if eid.startswith("sensor.") and eid.endswith(DEVICE_TYPE_SENSOR_SUFFIX):
                    candidates.append(eid)
        except Exception:  # pragma: no cover
            candidates = []

    if not candidates:
        suffix = DEVICE_TYPE_SENSOR_SUFFIX
        for state in hass.states.async_all("sensor"):
            eid = str(state.entity_id)
            if eid.endswith(suffix):
                candidates.append(eid)
        # Prefer the sensor sharing the fan's IP token (siku naming).
        ip = _extract_ip(fan_entity_id)
        if ip:
            candidates.sort(key=lambda eid: 0 if ip in eid else 1)
        same_device_only = False

    for eid in candidates:
        state = hass.states.get(eid)
        if state is None:
            continue
        value = _to_int(state.state)
        if value is not None:
            return value
        if same_device_only:
            # Registry says this is the right device but the sensor is
            # unavailable; do not fall through to unrelated sensors.
            return None
    return None


def _extract_ip(entity_id: str) -> Optional[str]:
    import re

    match = re.search(r"(\d{1,3}_\d{1,3}_\d{1,3}_\d{1,3})", entity_id)
    return match.group(1) if match else None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def zone_airflow_semantic(zone: Dict[str, Any]) -> str:
    """Return the zone's airflow semantic with a safe default."""
    return normalize_semantic_airflow(zone.get("airflow_mode")) or AIRFLOW_NONE


def build_zone_airflow_plans(
    hass: Any,
    outputs: Iterable[str],
    semantic: str,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[OutputAirflowPlan]:
    plans: List[OutputAirflowPlan] = []
    seen = set()
    for entity_id in outputs or []:
        entity_id = str(entity_id or "").strip()
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        plans.append(resolve_output_airflow(hass, entity_id, semantic, overrides))
    return plans
