from __future__ import annotations

from typing import Any, Mapping, TypedDict

from .constants import (
    GROUP_MISSING_POLICY_AS_LEVEL,
    GROUP_MISSING_POLICY_DROP_ROWS,
)
from .defaults import DEFAULT_ANALYSIS_CONFIG, DEFAULT_PROFILE_RULES


class ProfileRules(TypedDict):
    categorical_max_cardinality: int
    datetime_categorical_max_cardinality: int
    text_unique_ratio_exclude: float
    numeric_parse_ratio: float
    datetime_parse_ratio: float


class AnalysisConfig(TypedDict):
    response: str
    primary_factors: list[str]
    group_variables: list[str]
    group_include_unobserved_combinations: bool
    group_missing_policy: str
    group_max_groups_analyzed: int
    group_min_primary_level_n: int
    categorical_covariates: list[str]
    continuous_covariates: list[str]
    forced_terms: list[str]
    interaction_candidates: list[str]
    max_covariates_in_model: int
    max_interactions_in_model: int
    max_total_models: int
    random_seed: int
    robust_se_mode: str
    include_robust_in_stability: bool
    vif_fail_threshold: float
    vif_warn_threshold: float
    bp_alpha: float
    shapiro_fail_alpha: float
    shapiro_warn_alpha: float
    cook_threshold_multiplier: float
    cook_fail: bool
    outlier_mad_threshold: float
    stable_sign_pct: float
    stable_practical_pct: float
    non_effect_pct: float
    engineering_delta: float
    equivalence_bound: float
    sign_flip_redflag_pct: float
    health_min_rows: int
    health_min_rows_per_group: int
    name: str


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "on", "yes"}:
        return True
    if text in {"0", "false", "off", "no"}:
        return False
    return default


def normalize_profile_rules(raw: Mapping[str, Any] | None = None) -> ProfileRules:
    src = dict(DEFAULT_PROFILE_RULES)
    if raw:
        src.update(dict(raw))

    rules: ProfileRules = {
        "categorical_max_cardinality": max(1, _to_int(src.get("categorical_max_cardinality"), DEFAULT_PROFILE_RULES["categorical_max_cardinality"])),
        "datetime_categorical_max_cardinality": max(
            1,
            _to_int(src.get("datetime_categorical_max_cardinality"), DEFAULT_PROFILE_RULES["datetime_categorical_max_cardinality"]),
        ),
        "text_unique_ratio_exclude": _to_float(src.get("text_unique_ratio_exclude"), DEFAULT_PROFILE_RULES["text_unique_ratio_exclude"]),
        "numeric_parse_ratio": _to_float(src.get("numeric_parse_ratio"), DEFAULT_PROFILE_RULES["numeric_parse_ratio"]),
        "datetime_parse_ratio": _to_float(src.get("datetime_parse_ratio"), DEFAULT_PROFILE_RULES["datetime_parse_ratio"]),
    }
    return rules


def normalize_analysis_config(
    raw: Mapping[str, Any] | None = None,
    *,
    base: Mapping[str, Any] | None = None,
) -> AnalysisConfig:
    src = dict(DEFAULT_ANALYSIS_CONFIG)
    if base:
        src.update(dict(base))
    if raw:
        src.update(dict(raw))

    cfg: AnalysisConfig = {
        "response": str(src.get("response", "")).strip(),
        "primary_factors": list(dict.fromkeys(as_str_list(src.get("primary_factors")))),
        "group_variables": list(dict.fromkeys(as_str_list(src.get("group_variables")))),
        "group_include_unobserved_combinations": _to_bool(src.get("group_include_unobserved_combinations"), False),
        "group_missing_policy": str(src.get("group_missing_policy", GROUP_MISSING_POLICY_AS_LEVEL)).strip().upper(),
        "group_max_groups_analyzed": max(1, _to_int(src.get("group_max_groups_analyzed"), DEFAULT_ANALYSIS_CONFIG["group_max_groups_analyzed"])),
        "group_min_primary_level_n": max(1, _to_int(src.get("group_min_primary_level_n"), DEFAULT_ANALYSIS_CONFIG["group_min_primary_level_n"])),
        "categorical_covariates": list(dict.fromkeys(as_str_list(src.get("categorical_covariates")))),
        "continuous_covariates": list(dict.fromkeys(as_str_list(src.get("continuous_covariates")))),
        "forced_terms": list(dict.fromkeys(as_str_list(src.get("forced_terms")))),
        "interaction_candidates": list(dict.fromkeys(as_str_list(src.get("interaction_candidates")))),
        "max_covariates_in_model": max(0, _to_int(src.get("max_covariates_in_model"), DEFAULT_ANALYSIS_CONFIG["max_covariates_in_model"])),
        "max_interactions_in_model": max(0, _to_int(src.get("max_interactions_in_model"), DEFAULT_ANALYSIS_CONFIG["max_interactions_in_model"])),
        "max_total_models": max(1, _to_int(src.get("max_total_models"), DEFAULT_ANALYSIS_CONFIG["max_total_models"])),
        "random_seed": _to_int(src.get("random_seed"), DEFAULT_ANALYSIS_CONFIG["random_seed"]),
        "robust_se_mode": str(src.get("robust_se_mode", DEFAULT_ANALYSIS_CONFIG["robust_se_mode"])).strip().upper() or "HC3",
        "include_robust_in_stability": _to_bool(src.get("include_robust_in_stability"), True),
        "vif_fail_threshold": _to_float(src.get("vif_fail_threshold"), DEFAULT_ANALYSIS_CONFIG["vif_fail_threshold"]),
        "vif_warn_threshold": _to_float(src.get("vif_warn_threshold"), DEFAULT_ANALYSIS_CONFIG["vif_warn_threshold"]),
        "bp_alpha": _to_float(src.get("bp_alpha"), DEFAULT_ANALYSIS_CONFIG["bp_alpha"]),
        "shapiro_fail_alpha": _to_float(src.get("shapiro_fail_alpha"), DEFAULT_ANALYSIS_CONFIG["shapiro_fail_alpha"]),
        "shapiro_warn_alpha": _to_float(src.get("shapiro_warn_alpha"), DEFAULT_ANALYSIS_CONFIG["shapiro_warn_alpha"]),
        "cook_threshold_multiplier": _to_float(
            src.get("cook_threshold_multiplier"), DEFAULT_ANALYSIS_CONFIG["cook_threshold_multiplier"]
        ),
        "cook_fail": _to_bool(src.get("cook_fail"), False),
        "outlier_mad_threshold": _to_float(src.get("outlier_mad_threshold"), DEFAULT_ANALYSIS_CONFIG["outlier_mad_threshold"]),
        "stable_sign_pct": _to_float(src.get("stable_sign_pct"), DEFAULT_ANALYSIS_CONFIG["stable_sign_pct"]),
        "stable_practical_pct": _to_float(src.get("stable_practical_pct"), DEFAULT_ANALYSIS_CONFIG["stable_practical_pct"]),
        "non_effect_pct": _to_float(src.get("non_effect_pct"), DEFAULT_ANALYSIS_CONFIG["non_effect_pct"]),
        "engineering_delta": _to_float(src.get("engineering_delta"), DEFAULT_ANALYSIS_CONFIG["engineering_delta"]),
        "equivalence_bound": _to_float(src.get("equivalence_bound"), DEFAULT_ANALYSIS_CONFIG["equivalence_bound"]),
        "sign_flip_redflag_pct": _to_float(src.get("sign_flip_redflag_pct"), DEFAULT_ANALYSIS_CONFIG["sign_flip_redflag_pct"]),
        "health_min_rows": max(1, _to_int(src.get("health_min_rows"), DEFAULT_ANALYSIS_CONFIG["health_min_rows"])),
        "health_min_rows_per_group": max(
            1, _to_int(src.get("health_min_rows_per_group"), DEFAULT_ANALYSIS_CONFIG["health_min_rows_per_group"])
        ),
        "name": str(src.get("name", "SME Analysis")).strip() or "SME Analysis",
    }
    if cfg["group_missing_policy"] not in {GROUP_MISSING_POLICY_AS_LEVEL, GROUP_MISSING_POLICY_DROP_ROWS}:
        cfg["group_missing_policy"] = GROUP_MISSING_POLICY_AS_LEVEL
    return cfg
