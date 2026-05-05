from __future__ import annotations

from typing import Dict, List


STAGE_SPECS: Dict[str, List[str]] = {
    "origin_city_static": [
        "outflow_push_strength",
        "labor_export_tendency",
        "cost_escape_pressure",
        "intercity_departure_readiness",
        "outward_business_linkage",
        "tourism_departure_propensity",
        "administrative_radiation_out",
        "transport_departure_support",
    ],
    "destination_city_static": [
        "inflow_pull_strength",
        "job_absorption_capacity",
        "service_absorption_capacity",
        "amenity_attractiveness",
        "tourism_reception_capacity",
        "policy_resource_concentration",
        "gateway_arrival_support",
        "settlement_attractiveness",
    ],
    "pair_relation_static": [
        "geographic_friction",
        "transport_connectivity",
        "administrative_barrier",
        "economic_complementarity",
        "functional_complementarity",
        "migration_cost",
        "accessibility_advantage",
    ],
}


STAGE_EXAMPLES: Dict[str, dict] = {
    "origin_city_static": {
        "outflow_push_strength": 3,
        "labor_export_tendency": 2,
        "cost_escape_pressure": 4,
        "intercity_departure_readiness": 3,
        "outward_business_linkage": 4,
        "tourism_departure_propensity": 2,
        "administrative_radiation_out": 3,
        "transport_departure_support": 4,
        "summary": "Strong outbound drivers and departure support.",
    },
    "destination_city_static": {
        "inflow_pull_strength": 4,
        "job_absorption_capacity": 4,
        "service_absorption_capacity": 3,
        "amenity_attractiveness": 3,
        "tourism_reception_capacity": 2,
        "policy_resource_concentration": 4,
        "gateway_arrival_support": 4,
        "settlement_attractiveness": 3,
        "summary": "Strong absorption capacity and arrival support.",
    },
    "pair_relation_static": {
        "geographic_friction": 1,
        "transport_connectivity": 4,
        "administrative_barrier": 1,
        "economic_complementarity": 3,
        "functional_complementarity": 3,
        "migration_cost": 1,
        "accessibility_advantage": 4,
        "summary": "Smooth connection, strong complementarity, and low impedance.",
    },
}


LEGACY_STAGE_ALIAS: Dict[str, str] = {
    "city_static": "origin_city_static",
    "pair_static": "pair_relation_static",
}

def normalize_stage_name(stage: str) -> str:
    return LEGACY_STAGE_ALIAS.get(stage, stage)


def stage_keys(stage: str) -> List[str]:
    stage = normalize_stage_name(stage)
    if stage not in STAGE_SPECS:
        raise KeyError(f"Unknown stage: {stage}")
    return STAGE_SPECS[stage]


def stage_example(stage: str) -> dict:
    stage = normalize_stage_name(stage)
    if stage not in STAGE_EXAMPLES:
        raise KeyError(f"Unknown stage: {stage}")
    return STAGE_EXAMPLES[stage]

def build_stage_output_example(stage: str) -> dict:
    base = stage_example(stage)
    result: Dict[str, object] = {}
    for key in stage_keys(stage):
        result[key] = int(base.get(key, 0))
        result[f"{key}_confidence"] = 0.80
        result[f"{key}_rationale"] = "Inferred from input attributes"
    result["summary"] = str(base.get("summary", ""))
    return result


def _safe_int_score(value: object) -> int:
    try:
        value = int(round(float(value)))
    except Exception:
        value = 0
    return max(0, min(4, value))


def _safe_confidence(value: object, fallback: float = 0.50) -> float:
    try:
        value = float(value)
    except Exception:
        value = fallback
    value = max(0.0, min(1.0, value))
    return round(value, 6)


def _safe_short_text(value: object, max_len: int = 80) -> str:
    text = str(value or "")
    text = text.replace("\n", " ").strip()
    return text[:max_len]


def sanitize_stage_result(stage: str, data: dict) -> dict:
    stage = normalize_stage_name(stage)
    keys = stage_keys(stage)
    cleaned: Dict[str, object] = {}

    for key in keys:
        cleaned[key] = _safe_int_score(data.get(key, 0))
        cleaned[f"{key}_confidence"] = _safe_confidence(data.get(f"{key}_confidence", 0.50))
        cleaned[f"{key}_rationale"] = _safe_short_text(data.get(f"{key}_rationale", ""), max_len=60)

    summary = _safe_short_text(data.get("summary", ""), max_len=160)
    cleaned["summary"] = summary
    return cleaned
