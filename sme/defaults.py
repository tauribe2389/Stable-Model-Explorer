from __future__ import annotations


DEFAULT_PROFILE_RULES = {
    "categorical_max_cardinality": 20,
    "datetime_categorical_max_cardinality": 31,
    "text_unique_ratio_exclude": 0.9,
    "numeric_parse_ratio": 0.95,
    "datetime_parse_ratio": 0.95,
}


DEFAULT_ANALYSIS_CONFIG = {
    "response": "",
    "primary_factors": [],
    "group_variables": [],
    "categorical_covariates": [],
    "continuous_covariates": [],
    "forced_terms": [],
    "interaction_candidates": [],
    "max_covariates_in_model": 6,
    "max_interactions_in_model": 3,
    "max_total_models": 2000,
    "random_seed": 13,
    "robust_se_mode": "HC3",
    "include_robust_in_stability": True,
    "vif_fail_threshold": 10.0,
    "vif_warn_threshold": 5.0,
    "bp_alpha": 0.05,
    "shapiro_fail_alpha": 0.01,
    "shapiro_warn_alpha": 0.05,
    "cook_threshold_multiplier": 4.0,
    "cook_fail": False,
    "outlier_mad_threshold": 3.5,
    "stable_sign_pct": 0.9,
    "stable_practical_pct": 0.8,
    "non_effect_pct": 0.8,
    "engineering_delta": 0.1,
    "equivalence_bound": 0.05,
    "sign_flip_redflag_pct": 0.6,
    "health_min_rows": 30,
    "health_min_rows_per_group": 5,
}
