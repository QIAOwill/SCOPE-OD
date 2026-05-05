from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .schemas import build_stage_output_example, stage_keys


COMMON_RULES = """
You are an expert in intercity mobility and origin-destination relation modeling. Your task is not to predict daily flow values. Your task is to convert structured static attributes into transferable and composable role-aware semantic priors. Return strict JSON only.

Rules:
1. Every primary semantic score must be an integer in {0, 1, 2, 3, 4}. A score of 0 means very low or weak; 4 means very high or strong.
2. For every semantic field, also return <field>_confidence and <field>_rationale.
3. The output should describe role-aware semantic priors, not generic city profiles or simple development-level rankings.
4. The same city may play asymmetric roles in different mobility contexts: origin fields describe departure-generation roles; destination fields describe arrival-attraction roles; pair-relation fields describe directional O-to-D relation tokens.
5. Do not equate stronger economy or larger population with higher scores for every field.
6. For pair-relation stages, interpret every relation directionally as O to D.
7. If evidence is weak or conflicting, lower the confidence but still provide the best-supported score.
8. Do not output any text outside the JSON object.
9. Keep the summary field concise.
""".strip()

ROLE_PRIOR_HINT = """
Scoring principles:
- Extract stable and transferable structural role priors rather than short-term shocks.
- Focus on functional position, behavioral tendency, and relation constraints in the intercity mobility system.
- Use economic structure, demographic structure, transport organization, administrative status, service resources, and functional mismatch as evidence.
- Avoid treating all fields as repeated measurements of the same latent dimension.
- Scores should reflect semantic disentanglement across roles.
""".strip()

TRANSFER_HINT = """
The downstream forecasting model will use these outputs as semantic priors and controller inputs. Prefer stable role semantics, cross-region transferable abstractions, and clearly composable semantic dimensions.
""".strip()

OUTPUT_SPEC_HINT = """
The JSON object must contain every required score field, every paired *_confidence field, every paired *_rationale field, and a summary. Do not add extra fields.
""".strip()

STAGE_FIELD_GUIDANCE: Dict[str, Dict[str, str]] = {
    "origin_city_static": {
        "outflow_push_strength": "Overall strength of structural drivers that generate outbound intercity movement.",
        "labor_export_tendency": "Tendency for labor or skills to move outward for employment.",
        "cost_escape_pressure": "Degree to which living, housing, or competition costs push residents outward.",
        "intercity_departure_readiness": "Maturity of networks and organization that enable intercity departures.",
        "outward_business_linkage": "Outbound business, project, and factor-mobility linkages.",
        "tourism_departure_propensity": "Propensity for non-settlement departures such as tourism and leisure.",
        "administrative_radiation_out": "Outward administrative or institutional radiation capacity.",
        "transport_departure_support": "Transport support for departure generation.",
    },
    "destination_city_static": {
        "inflow_pull_strength": "Overall structural attraction for arrivals and in-migration.",
        "job_absorption_capacity": "Capacity to absorb incoming people through jobs and industries.",
        "service_absorption_capacity": "Capacity of public and commercial services to receive arrivals.",
        "amenity_attractiveness": "Attractiveness from amenities, convenience, consumption, and urban experience.",
        "tourism_reception_capacity": "Capacity to receive short-term tourism and visits.",
        "policy_resource_concentration": "Attraction from administrative status and concentrated policy resources.",
        "gateway_arrival_support": "Support from gateway nodes and multimodal arrival infrastructure.",
        "settlement_attractiveness": "Attraction for long-term residence and settlement.",
    },
    "pair_relation_static": {
        "geographic_friction": "Spatial distance and geographic impedance in the O-to-D direction.",
        "transport_connectivity": "Transport smoothness and connectivity from O to D.",
        "administrative_barrier": "Barrier from administrative boundaries or institutional separation.",
        "economic_complementarity": "Complementarity in economic scale, income gradient, or market hierarchy.",
        "functional_complementarity": "Complementarity or mismatch in industrial, POI, and population functions.",
        "migration_cost": "Comprehensive cost of cross-city movement or settlement from O to D.",
        "accessibility_advantage": "Directional accessibility advantage from O to D.",
    },
}

STAGE_TASK_HINT: Dict[str, str] = {
    "origin_city_static": "Model the city as an origin-role semantic agent and focus on how it generates intercity departures.",
    "destination_city_static": "Model the city as a destination-role semantic agent and focus on how it receives and attracts arrivals.",
    "pair_relation_static": "Model the directed city pair as an O-to-D relation token and focus on complementarity, impedance, and accessibility.",
}

STAGE_FOCUS_HINT: Dict[str, List[str]] = {
    "origin_city_static": [
        "Whether outbound movement can be persistently activated.",
        "Whether employment, cost pressure, business linkage, or administrative radiation creates distinct departure drivers.",
        "Whether transport and network conditions convert latent demand into intercity departures.",
    ],
    "destination_city_static": [
        "Whether attraction comes from jobs, public services, amenities, gateway functions, or policy resources.",
        "Whether short-term visits and long-term settlement should be separated.",
        "Whether arrival support and settlement capacity coexist or diverge.",
    ],
    "pair_relation_static": [
        "Whether O-to-D distance, connectivity, and administrative boundaries create constraints or advantages.",
        "Whether economic gradients and functional mismatch can support persistent interaction.",
        "Whether migration cost and arrival convenience diverge and need separate scoring.",
    ],
}


def _format_json_example(stage: str) -> str:
    return json.dumps(build_stage_output_example(stage), ensure_ascii=False, indent=2)


def _format_required_output_desc(stage: str) -> str:
    lines = []
    for key in stage_keys(stage):
        lines.append(f"- {key}")
        lines.append(f"- {key}_confidence")
        lines.append(f"- {key}_rationale")
    lines.append("- summary")
    return "\n".join(lines)


def _format_field_guidance(stage: str) -> str:
    return "\n".join(f"- {key}: {STAGE_FIELD_GUIDANCE[stage][key]}" for key in stage_keys(stage))


def _format_focus(stage: str) -> str:
    return "\n".join(f"- {item}" for item in STAGE_FOCUS_HINT[stage])


def _build_system_prompt(stage: str, stage_instruction: str) -> str:
    return "\n".join([
        COMMON_RULES,
        ROLE_PRIOR_HINT,
        TRANSFER_HINT,
        OUTPUT_SPEC_HINT,
        STAGE_TASK_HINT[stage],
        stage_instruction,
    ])


def origin_city_static_prompts(record: Dict) -> Tuple[str, str]:
    stage = "origin_city_static"
    system = _build_system_prompt(stage, "Use city-level static attributes to output origin-role semantic JSON.")
    user = f"""
Return JSON scores for the static origin-role semantics of this city.

Urban agglomeration: {record['urban_agglomeration']}
City ID: {record['city_id']}
City name: {record['city_name']}

Task:
{STAGE_TASK_HINT[stage]}

Focus:
{_format_focus(stage)}

Field guidance:
{_format_field_guidance(stage)}

Core attributes:
- GDP: {record['gdp']}
- GDP per capita: {record['gdp_per_capita']}
- Income per capita: {record['income_per_capita']}
- Fixed asset investment: {record['fixed_asset_investment']}
- Population: {record['population']}
- Population density: {record['population_density']}
- Tertiary industry ratio: {record['tertiary_industry_ratio']}
- Secondary industry ratio: {record['secondary_industry_ratio']}
- Primary industry ratio: {record['primary_industry_ratio']}
- Road density: {record['road_density']}
- Road area per capita: {record['road_area_per_capita']}
- POI transport ratio: {record['poi_transport_ratio']}
- POI company ratio: {record['poi_company_ratio']}
- POI tourism ratio: {record['poi_tourism_ratio']}
- POI density: {record['poi_density']}
- Reachable cities within 200 km: {record['reachable_city_200km']}
- Reachable cities within 500 km: {record['reachable_city_500km']}
- Reachable cities within 1000 km: {record['reachable_city_1000km']}
- Dialect diversity index: {record['dialect_diversity_index']}
- Average wage: {record['avg_wage_latest']}
- House price: {record['house_price_latest']}
- Scenic spot count: {record['scenic_spot_count']}
- A-level scenic spot count: {record['a_level_scenic_count']}
- Provincial capital flag: {record['is_provincial_capital']}
- Municipality flag: {record['is_municipality']}

Required fields:
{_format_required_output_desc(stage)}

Example:
{_format_json_example(stage)}
""".strip()
    return system, user


def destination_city_static_prompts(record: Dict) -> Tuple[str, str]:
    stage = "destination_city_static"
    system = _build_system_prompt(stage, "Use city-level static attributes to output destination-role semantic JSON.")
    user = f"""
Return JSON scores for the static destination-role semantics of this city.

Urban agglomeration: {record['urban_agglomeration']}
City ID: {record['city_id']}
City name: {record['city_name']}

Task:
{STAGE_TASK_HINT[stage]}

Focus:
{_format_focus(stage)}

Field guidance:
{_format_field_guidance(stage)}

Core attributes:
- GDP: {record['gdp']}
- GDP per capita: {record['gdp_per_capita']}
- Income per capita: {record['income_per_capita']}
- Fixed asset investment: {record['fixed_asset_investment']}
- Population: {record['population']}
- Population density: {record['population_density']}
- Tertiary industry ratio: {record['tertiary_industry_ratio']}
- Secondary industry ratio: {record['secondary_industry_ratio']}
- Primary industry ratio: {record['primary_industry_ratio']}
- Road density: {record['road_density']}
- Road area per capita: {record['road_area_per_capita']}
- POI transport ratio: {record['poi_transport_ratio']}
- POI company ratio: {record['poi_company_ratio']}
- POI tourism ratio: {record['poi_tourism_ratio']}
- POI density: {record['poi_density']}
- Reachable cities within 200 km: {record['reachable_city_200km']}
- Reachable cities within 500 km: {record['reachable_city_500km']}
- Reachable cities within 1000 km: {record['reachable_city_1000km']}
- Dialect diversity index: {record['dialect_diversity_index']}
- Average wage: {record['avg_wage_latest']}
- House price: {record['house_price_latest']}
- Scenic spot count: {record['scenic_spot_count']}
- A-level scenic spot count: {record['a_level_scenic_count']}
- Provincial capital flag: {record['is_provincial_capital']}
- Municipality flag: {record['is_municipality']}

Required fields:
{_format_required_output_desc(stage)}

Example:
{_format_json_example(stage)}
""".strip()
    return system, user


def pair_relation_static_prompts(record: Dict) -> Tuple[str, str]:
    stage = "pair_relation_static"
    system = _build_system_prompt(stage, "Use OD-pair static attributes to output directed relation-semantic JSON.")
    user = f"""
Return JSON scores for the static relation token and impedance semantics in the O-to-D direction.

Urban agglomeration: {record['urban_agglomeration']}
Origin O: {record['origin_city']} ({record['origin_id']})
Destination D: {record['destination_city']} ({record['destination_id']})

Task:
{STAGE_TASK_HINT[stage]}

Focus:
{_format_focus(stage)}

Field guidance:
{_format_field_guidance(stage)}

Core attributes:
- Line distance: {record['distance_line']}
- Road distance: {record['distance_road']}
- Railway distance: {record['distance_railway']}
- Adjacent flag: {record['is_adjacent']}
- Same province flag: {record['same_province']}
- GDP gap: {record['gdp_gap']}
- Income gap: {record['income_gap']}
- Population gap: {record['population_gap']}
- Industry-structure similarity: {record['industry_structure_similarity']}
- POI-structure similarity: {record['poi_structure_similarity']}
- Direct HSR flag: {record['hsr_direct_flag']}
- HSR train count: {record['hsr_train_count']}
- HSR minimum travel time: {record['hsr_min_travel_time']}
- HSR average travel time: {record['hsr_avg_travel_time']}
- HSR service intensity: {record['hsr_service_intensity']}

Required fields:
{_format_required_output_desc(stage)}

Example:
{_format_json_example(stage)}
""".strip()
    return system, user
