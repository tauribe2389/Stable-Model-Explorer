from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from .db import fetchall, fetchone, from_json, get_conn, to_json, utcnow_iso
from .profiling import excluded_row_indices, load_dataset_dataframe

matplotlib.use("Agg")


def quote_name(name: str) -> str:
    escaped = name.replace('"', '\\"')
    return f'Q("{escaped}")'


def term_expr(name: str, categorical_names: set[str]) -> str:
    q = quote_name(name)
    return f"C({q})" if name in categorical_names else q


def interaction_expr(interaction: str, categorical_names: set[str]) -> str:
    parts = [p.strip() for p in interaction.split(":") if p.strip()]
    if len(parts) != 2:
        return ""
    left, right = parts
    return f"{term_expr(left, categorical_names)}:{term_expr(right, categorical_names)}"


def _sanitize_selection(names: list[str], all_columns: set[str]) -> list[str]:
    return [n for n in names if n in all_columns]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def generate_model_registry(
    db_path: str,
    analysis_id: int,
    config: dict[str, Any],
    columns_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    all_cols = {c["column_name"] for c in columns_meta}
    role_by_col = {c["column_name"]: c["model_role"] for c in columns_meta}
    categorical_names = {c["column_name"] for c in columns_meta if c["model_role"] == "categorical"}

    response = config["response"]
    if response not in all_cols:
        raise ValueError("Response column missing from dataset.")

    primary = _sanitize_selection(_as_list(config.get("primary_factors")), all_cols)
    forced_terms = _sanitize_selection(_as_list(config.get("forced_terms")), all_cols)
    forced = list(dict.fromkeys(primary + forced_terms))
    categorical_cov = _sanitize_selection(_as_list(config.get("categorical_covariates")), all_cols)
    continuous_cov = _sanitize_selection(_as_list(config.get("continuous_covariates")), all_cols)
    optional_terms = [x for x in dict.fromkeys(categorical_cov + continuous_cov) if x not in forced and x != response]
    interaction_candidates = [
        i
        for i in _as_list(config.get("interaction_candidates"))
        if all(p.strip() in all_cols for p in i.split(":"))
    ]

    max_cov = max(0, int(config.get("max_covariates_in_model", 6)))
    max_inter = max(0, int(config.get("max_interactions_in_model", 3)))
    max_total = max(1, int(config.get("max_total_models", 2000)))
    seed = int(config.get("random_seed", 13))
    rng = random.Random(seed)

    combos: list[dict[str, Any]] = []
    optional_limit = min(len(optional_terms), max_cov)
    for k in range(0, optional_limit + 1):
        for subset in itertools.combinations(optional_terms, k):
            terms = list(dict.fromkeys(forced + list(subset)))
            if not terms:
                continue
            available_interactions = []
            for inter in interaction_candidates:
                parts = [p.strip() for p in inter.split(":")]
                if len(parts) != 2:
                    continue
                if parts[0] in terms and parts[1] in terms:
                    available_interactions.append(inter)
            inter_subsets = [tuple()]
            if available_interactions and max_inter > 0:
                for ik in range(1, min(max_inter, len(available_interactions)) + 1):
                    inter_subsets.extend(itertools.combinations(available_interactions, ik))
            for inter_subset in inter_subsets:
                expr_terms = [term_expr(t, categorical_names) for t in terms]
                inter_exprs = [interaction_expr(i, categorical_names) for i in inter_subset if interaction_expr(i, categorical_names)]
                rhs = " + ".join(expr_terms + inter_exprs)
                formula = f"{quote_name(response)} ~ {rhs}"
                has_cont = any(role_by_col.get(t) == "continuous" for t in terms if t != response)
                model_class = "ANCOVA" if has_cont else "ANOVA"
                combos.append(
                    {
                        "formula": formula,
                        "model_class": model_class,
                        "included_terms": terms,
                        "interactions": list(inter_subset),
                    }
                )

    if not combos:
        baseline_terms = primary or [x for x in forced if x != response]
        if baseline_terms:
            formula = f'{quote_name(response)} ~ {" + ".join(term_expr(t, categorical_names) for t in baseline_terms)}'
            combos = [
                {
                    "formula": formula,
                    "model_class": "ANOVA",
                    "included_terms": baseline_terms,
                    "interactions": [],
                }
            ]

    if len(combos) > max_total:
        sampled = rng.sample(combos, max_total)
        sampled.sort(key=lambda x: x["formula"])
        combos = sampled
    else:
        combos.sort(key=lambda x: x["formula"])

    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM model_registry WHERE analysis_id = ?", [analysis_id])
        conn.executemany(
            """
            INSERT INTO model_registry(
                analysis_id, model_idx, formula, model_class, included_terms_json, interactions_json, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
            """,
            [
                [
                    analysis_id,
                    idx + 1,
                    combo["formula"],
                    combo["model_class"],
                    to_json(combo["included_terms"]),
                    to_json(combo["interactions"]),
                    utcnow_iso(),
                ]
                for idx, combo in enumerate(combos)
            ],
        )
        conn.commit()
    return combos


def load_registry(db_path: str, analysis_id: int) -> list[dict[str, Any]]:
    rows = fetchall(
        db_path,
        "SELECT * FROM model_registry WHERE analysis_id = ? ORDER BY model_idx",
        [analysis_id],
    )
    out = []
    for r in rows:
        rec = dict(r)
        rec["included_terms"] = from_json(rec["included_terms_json"], [])
        rec["interactions"] = from_json(rec["interactions_json"], [])
        out.append(rec)
    return out


def _prepare_dataframe_for_model(
    df: pd.DataFrame,
    columns_meta: list[dict[str, Any]],
    response: str,
    terms: list[str],
    excluded_rows: set[int],
) -> pd.DataFrame:
    role_by_col = {c["column_name"]: c["model_role"] for c in columns_meta}
    inferred_by_col = {c["column_name"]: c["inferred_type"] for c in columns_meta}
    keep_cols = ["row_index", response] + [t for t in terms if t != response]
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in df.columns]
    model_df = df[keep_cols].copy()
    if excluded_rows:
        model_df = model_df[~model_df["row_index"].isin(excluded_rows)]

    for col in keep_cols:
        if col == "row_index":
            continue
        role = role_by_col.get(col)
        inferred = inferred_by_col.get(col)
        if col == response or role == "continuous":
            if inferred == "datetime":
                dt = pd.to_datetime(model_df[col], errors="coerce", utc=True)
                model_df[col] = dt.view("int64") / (24 * 60 * 60 * 1_000_000_000)
            else:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        elif role == "categorical":
            if inferred == "datetime":
                dt = pd.to_datetime(model_df[col], errors="coerce", utc=True)
                model_df[col] = dt.dt.strftime("%Y-%m-%d")
            else:
                model_df[col] = model_df[col].astype(str)
        else:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df = model_df.dropna()
    return model_df


def _compute_vif(exog: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    out = []
    if exog.shape[1] <= 1:
        return out
    for i in range(exog.shape[1]):
        if names[i].lower() == "intercept":
            continue
        try:
            vif = float(variance_inflation_factor(exog, i))
        except Exception:
            vif = float("inf")
        out.append({"term": names[i], "vif": vif})
    return out


def _save_diagnostic_plots(
    artifact_dir: str,
    analysis_id: int,
    model_idx: int,
    fit,
) -> dict[str, str]:
    out_dir = Path(artifact_dir) / "plots" / f"analysis_{analysis_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resid_path = out_dir / f"model_{model_idx}_resid_vs_fit.png"
    qq_path = out_dir / f"model_{model_idx}_qq.png"

    fitted = fit.fittedvalues
    resid = fit.resid

    plt.figure(figsize=(6, 4))
    plt.scatter(fitted, resid, alpha=0.6, s=18)
    plt.axhline(0, color="red", linewidth=1)
    plt.xlabel("Fitted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(resid_path)
    plt.close()

    plt.figure(figsize=(5, 5))
    sm.qqplot(resid, line="45", fit=True)
    plt.title("Q-Q Plot")
    plt.tight_layout()
    plt.savefig(qq_path)
    plt.close()

    return {"resid_vs_fit": str(resid_path), "qq_plot": str(qq_path)}


def _coef_dict_from_fit(fit, robust_mode: str | None, apply_robust: bool) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    if robust_mode and apply_robust:
        robust_fit = fit.get_robustcov_results(cov_type=robust_mode)
        coeffs = {k: float(v) for k, v in zip(fit.params.index, robust_fit.params)}
        pvals = {k: float(v) for k, v in zip(fit.params.index, robust_fit.pvalues)}
        ci_np = robust_fit.conf_int()
        ci = {k: [float(ci_np[i][0]), float(ci_np[i][1])] for i, k in enumerate(fit.params.index)}
        return coeffs, pvals, ci
    coeffs = {k: float(v) for k, v in fit.params.items()}
    pvals = {k: float(v) for k, v in fit.pvalues.items()}
    ci_df = fit.conf_int()
    ci = {idx: [float(row[0]), float(row[1])] for idx, row in ci_df.iterrows()}
    return coeffs, pvals, ci


def _extract_lsmeans(
    fit,
    model_df: pd.DataFrame,
    factors: list[str],
    columns_meta: list[dict[str, Any]],
    response: str,
) -> dict[str, Any]:
    role_by_col = {c["column_name"]: c["model_role"] for c in columns_meta}
    lsmeans: dict[str, Any] = {}
    predictors = [c for c in model_df.columns if c not in {"row_index", response}]
    for factor in factors:
        if factor not in model_df.columns:
            continue
        levels = [x for x in sorted(model_df[factor].dropna().unique().tolist()) if str(x) != ""]
        if len(levels) < 2:
            continue
        base: dict[str, Any] = {}
        for col in predictors:
            if role_by_col.get(col) == "continuous":
                base[col] = float(pd.to_numeric(model_df[col], errors="coerce").mean())
            else:
                mode = model_df[col].mode(dropna=True)
                base[col] = str(mode.iloc[0]) if len(mode) else ""

        means = []
        for lvl in levels:
            row = dict(base)
            row[factor] = lvl
            pred_df = pd.DataFrame([row])
            try:
                estimate = float(fit.predict(pred_df).iloc[0])
            except Exception:
                continue
            means.append({"level": str(lvl), "adjusted_mean": estimate})
        means.sort(key=lambda x: x["adjusted_mean"], reverse=True)
        pairwise = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                a = means[i]
                b = means[j]
                pairwise.append(
                    {
                        "level_a": a["level"],
                        "level_b": b["level"],
                        "difference": float(a["adjusted_mean"] - b["adjusted_mean"]),
                    }
                )
        lsmeans[factor] = {"adjusted_means": means, "pairwise_differences": pairwise}
    return lsmeans


def run_registry_models(
    db_path: str,
    artifact_dir: str,
    analysis_id: int,
    config: dict[str, Any],
    columns_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis:
        raise ValueError(f"Unknown analysis_id={analysis_id}")
    dataset_id = int(analysis["dataset_id"])
    df = load_dataset_dataframe(db_path, dataset_id)
    excluded_rows = excluded_row_indices(db_path, analysis_id)
    registry = load_registry(db_path, analysis_id)
    robust_mode = str(config.get("robust_se_mode", "")).strip().upper()
    use_robust = robust_mode in {"HC0", "HC1", "HC2", "HC3"}
    response = config["response"]
    primary_factors = _as_list(config.get("primary_factors"))
    shapiro_fail_alpha = float(config.get("shapiro_fail_alpha", 0.01))
    shapiro_warn_alpha = float(config.get("shapiro_warn_alpha", 0.05))
    bp_alpha = float(config.get("bp_alpha", 0.05))
    vif_fail = float(config.get("vif_fail_threshold", 10))
    vif_warn = float(config.get("vif_warn_threshold", 5))
    cook_mul = float(config.get("cook_threshold_multiplier", 4.0))
    cook_fail = bool(config.get("cook_fail", False))

    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM model_runs WHERE analysis_id = ?", [analysis_id])
        conn.commit()

    counts = {"total": len(registry), "valid": 0, "invalid": 0, "valid_with_robust_se": 0}
    for reg in registry:
        invalid_reasons: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        coeffs: dict[str, float] = {}
        pvals: dict[str, float] = {}
        ci: dict[str, list[float]] = {}
        anova_table: list[dict[str, Any]] = []
        lsmeans: dict[str, Any] = {}
        cooks_payload: list[dict[str, Any]] = []
        residual_diag: dict[str, Any] = {}
        artifacts: dict[str, str] = {}
        n_obs = 0
        status = "completed"
        validity_class = "invalid"
        try:
            terms = reg["included_terms"]
            model_df = _prepare_dataframe_for_model(df, columns_meta, response, terms, excluded_rows)
            n_obs = int(len(model_df))
            if n_obs < 8:
                invalid_reasons.append("Insufficient rows after exclusions and NA drop.")
                raise ValueError("Insufficient rows")

            fit = smf.ols(reg["formula"], data=model_df).fit()
            residuals = fit.resid
            exog = fit.model.exog
            exog_names = fit.model.exog_names

            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
                metrics["bp_stat"] = float(bp_stat)
                metrics["bp_p"] = float(bp_p)
            except Exception as exc:
                warnings.append(f"Breusch-Pagan unavailable: {exc}")
                bp_p = 1.0
                metrics["bp_p"] = 1.0

            try:
                resid_arr = np.asarray(residuals)
                if len(resid_arr) <= 5000:
                    sample = resid_arr
                else:
                    rng = np.random.RandomState(13)
                    sample = rng.choice(resid_arr, size=5000, replace=False)
                shap_stat, shap_p = stats.shapiro(sample)
                metrics["shapiro_stat"] = float(shap_stat)
                metrics["shapiro_p"] = float(shap_p)
                if shap_p < shapiro_fail_alpha:
                    warnings.append("Residual normality likely violated (Shapiro below fail alpha).")
                elif shap_p < shapiro_warn_alpha:
                    warnings.append("Residual normality warning (Shapiro below warn alpha).")
            except Exception as exc:
                warnings.append(f"Shapiro unavailable: {exc}")

            vifs = _compute_vif(exog, exog_names)
            metrics["vif"] = vifs
            max_vif = max((v["vif"] for v in vifs), default=0.0)
            metrics["max_vif"] = float(max_vif)
            if max_vif >= vif_fail:
                invalid_reasons.append(f"VIF exceeds fail threshold ({max_vif:.2f} >= {vif_fail}).")
            elif max_vif >= vif_warn:
                warnings.append(f"VIF warning ({max_vif:.2f} >= {vif_warn}).")

            influence = OLSInfluence(fit)
            cooks = influence.cooks_distance[0]
            cook_threshold = cook_mul / max(1, n_obs)
            high_mask = cooks > cook_threshold
            high_count = int(np.sum(high_mask))
            metrics["cook_threshold"] = float(cook_threshold)
            metrics["max_cook"] = float(np.max(cooks)) if len(cooks) else 0.0
            metrics["high_cook_count"] = high_count
            if high_count > 0:
                msg = f"{high_count} points above Cook threshold {cook_threshold:.4f}"
                if cook_fail:
                    invalid_reasons.append(msg)
                else:
                    warnings.append(msg)
            cooks_payload = [
                {
                    "row_index": int(row_idx),
                    "cooks_distance": float(cook),
                    "high_leverage_flag": bool(cook > cook_threshold),
                }
                for row_idx, cook in zip(model_df["row_index"], cooks)
            ]

            if bp_p < bp_alpha:
                if use_robust:
                    warnings.append("Heteroscedasticity detected; robust SE applied.")
                else:
                    invalid_reasons.append("Breusch-Pagan indicates heteroscedasticity.")

            coeffs, pvals, ci = _coef_dict_from_fit(fit, robust_mode if use_robust else None, apply_robust=use_robust)
            metrics["r2"] = float(fit.rsquared)
            metrics["adj_r2"] = float(fit.rsquared_adj)
            metrics["aic"] = float(fit.aic)
            metrics["bic"] = float(fit.bic)

            try:
                anova_df = sm.stats.anova_lm(fit, typ=2)
                anova_table = [
                    {
                        "term": str(idx),
                        "sum_sq": float(row["sum_sq"]) if "sum_sq" in row else None,
                        "df": float(row["df"]) if "df" in row else None,
                        "f": float(row["F"]) if "F" in row and not pd.isna(row["F"]) else None,
                        "p": float(row["PR(>F)"]) if "PR(>F)" in row and not pd.isna(row["PR(>F)"]) else None,
                    }
                    for idx, row in anova_df.iterrows()
                ]
            except Exception as exc:
                warnings.append(f"ANOVA table unavailable: {exc}")

            lsmeans = _extract_lsmeans(fit, model_df, primary_factors, columns_meta, response)
            residual_diag = {
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
            }
            artifacts = _save_diagnostic_plots(artifact_dir, analysis_id, int(reg["model_idx"]), fit)

            if invalid_reasons:
                validity_class = "invalid"
            elif bp_p < bp_alpha and use_robust:
                validity_class = "valid_with_robust_se"
            else:
                validity_class = "valid"
        except Exception as exc:
            if not invalid_reasons:
                invalid_reasons.append(f"Model fit failed: {exc}")
            status = "failed"
            validity_class = "invalid"

        if validity_class == "valid":
            counts["valid"] += 1
        elif validity_class == "valid_with_robust_se":
            counts["valid_with_robust_se"] += 1
        else:
            counts["invalid"] += 1

        with get_conn(db_path) as conn:
            conn.execute(
                """
                INSERT INTO model_runs(
                    analysis_id, registry_id, run_at, status, validity_class, invalid_reasons_json, warnings_json,
                    metrics_json, coeffs_json, pvalues_json, ci_json, anova_json, lsmeans_json, cooks_json,
                    residual_diag_json, artifacts_json, n_obs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    analysis_id,
                    reg["id"],
                    utcnow_iso(),
                    status,
                    validity_class,
                    to_json(invalid_reasons),
                    to_json(warnings),
                    to_json(metrics),
                    to_json(coeffs),
                    to_json(pvals),
                    to_json(ci),
                    to_json(anova_table),
                    to_json(lsmeans),
                    to_json(cooks_payload),
                    to_json(residual_diag),
                    to_json(artifacts),
                    n_obs,
                ],
            )
            conn.execute("UPDATE model_registry SET status = 'completed' WHERE id = ?", [reg["id"]])
            conn.execute("UPDATE analyses SET status = 'models_ran', updated_at = ? WHERE id = ?", [utcnow_iso(), analysis_id])
            conn.commit()
    return counts


def list_model_runs(db_path: str, analysis_id: int) -> list[dict[str, Any]]:
    rows = fetchall(
        db_path,
        """
        SELECT mr.*, rg.model_idx, rg.formula, rg.model_class, rg.included_terms_json, rg.interactions_json
        FROM model_runs mr
        JOIN model_registry rg ON rg.id = mr.registry_id
        WHERE mr.analysis_id = ?
        ORDER BY rg.model_idx
        """,
        [analysis_id],
    )
    out = []
    for r in rows:
        rec = dict(r)
        rec["invalid_reasons"] = from_json(rec["invalid_reasons_json"], [])
        rec["warnings"] = from_json(rec["warnings_json"], [])
        rec["metrics"] = from_json(rec["metrics_json"], {})
        rec["coeffs"] = from_json(rec["coeffs_json"], {})
        rec["pvalues"] = from_json(rec["pvalues_json"], {})
        rec["ci"] = from_json(rec["ci_json"], {})
        rec["anova"] = from_json(rec["anova_json"], [])
        rec["lsmeans"] = from_json(rec["lsmeans_json"], {})
        rec["cooks"] = from_json(rec["cooks_json"], [])
        rec["artifacts"] = from_json(rec["artifacts_json"], {})
        rec["included_terms"] = from_json(rec["included_terms_json"], [])
        rec["interactions"] = from_json(rec["interactions_json"], [])
        out.append(rec)
    return out


def _estimate_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _stratified_metrics(values_by_class: dict[str, list[float]], pvals_by_class: dict[str, list[float]], thresholds: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cls, vals in values_by_class.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        pos = float(np.mean(arr > 0))
        neg = float(np.mean(arr < 0))
        sign_consistency = max(pos, neg)
        practical_rate = float(np.mean(np.abs(arr) > thresholds["engineering_delta"]))
        equivalence_rate = float(np.mean(np.abs(arr) < thresholds["equivalence_bound"]))
        sig_vals = pvals_by_class.get(cls, [])
        p_sig_rate = float(np.mean(np.array(sig_vals) < 0.05)) if sig_vals else None
        out[cls] = {
            "estimate_distribution": _estimate_distribution(vals),
            "sign_consistency": sign_consistency,
            "practical_rate": practical_rate,
            "equivalence_rate": equivalence_rate,
            "p_sig_rate": p_sig_rate,
            "n_models": len(vals),
        }
    return out


def _effect_sensitivity(effect_model_values: list[tuple[float, list[str]]], candidate_covariates: list[str], delta: float) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    if len(effect_model_values) < 6:
        return flags
    for cov in candidate_covariates:
        in_vals = [v for v, terms in effect_model_values if cov in terms]
        out_vals = [v for v, terms in effect_model_values if cov not in terms]
        if len(in_vals) < 2 or len(out_vals) < 2:
            continue
        med_in = float(np.median(in_vals))
        med_out = float(np.median(out_vals))
        sign_flip = (med_in > 0 and med_out < 0) or (med_in < 0 and med_out > 0)
        large_shift = abs(med_in - med_out) > delta
        flags[f"depends_on_covariate:{cov}"] = bool(sign_flip or large_shift)
    return {k: v for k, v in flags.items() if v}


def _bucket_effect(
    values: list[float],
    practical_threshold: float,
    equivalence_bound: float,
    stable_sign_pct: float,
    stable_practical_pct: float,
    non_effect_pct: float,
    redflag_sign_pct: float,
    has_sensitivity: bool,
) -> tuple[str, float, float, float]:
    if not values:
        return "NON_EFFECT", 1.0, 0.0, 1.0
    arr = np.array(values, dtype=float)
    pos_rate = float(np.mean(arr > 0))
    neg_rate = float(np.mean(arr < 0))
    sign_consistency = max(pos_rate, neg_rate)
    practical_rate = float(np.mean(np.abs(arr) > practical_threshold))
    equivalence_rate = float(np.mean(np.abs(arr) < equivalence_bound))
    has_sign_flip = pos_rate > 0 and neg_rate > 0

    if has_sign_flip and sign_consistency < redflag_sign_pct:
        return "REDFLAG", sign_consistency, practical_rate, equivalence_rate
    if equivalence_rate >= non_effect_pct:
        return "NON_EFFECT", sign_consistency, practical_rate, equivalence_rate
    if sign_consistency >= stable_sign_pct and practical_rate >= stable_practical_pct and not has_sensitivity:
        return "STABLE", sign_consistency, practical_rate, equivalence_rate
    return "CONDITIONAL", sign_consistency, practical_rate, equivalence_rate


def _safe_slug(text: str, max_len: int = 120) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(text))
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if not clean:
        clean = "effect"
    return clean[:max_len]


def _stability_plot_dir(artifact_dir: str, analysis_id: int) -> Path:
    out_dir = Path(artifact_dir) / "plots" / f"analysis_{analysis_id}" / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_effect_distribution_plot(
    out_dir: Path,
    effect_id: str,
    values: list[float],
    x_label: str,
) -> str | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    bins = min(36, max(8, int(np.sqrt(len(arr)) * 2)))
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(arr, bins=bins, color="#4b8bd6", alpha=0.78, edgecolor="#ffffff")
    ax.axvline(0.0, color="#9f1239", linestyle="--", linewidth=1.4, label="0 reference")
    ax.axvline(median, color="#0f172a", linewidth=1.3, label="Median")
    if q3 > q1:
        ax.axvspan(float(q1), float(q3), color="#93c5fd", alpha=0.25, label="IQR")
    sign_flip = bool(np.any(arr > 0) and np.any(arr < 0))
    title_suffix = "sign flips present" if sign_flip else "one-sided sign"
    ax.set_title(f"{effect_id}\n{title_suffix}", fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    filename = f"{_safe_slug(effect_id)}_distribution.png"
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_rank_stability_plot(
    out_dir: Path,
    factor_name: str,
    level_to_ranks: dict[str, list[float]],
) -> str | None:
    if not level_to_ranks:
        return None
    ordered = sorted(level_to_ranks.items(), key=lambda kv: float(np.median(kv[1])) if kv[1] else 9999.0)
    labels = [str(k) for k, _ in ordered]
    data = [list(map(float, vals)) for _, vals in ordered]
    if not any(data):
        return None

    fig, ax = plt.subplots(figsize=(max(6.2, 1.2 * len(labels)), 4.0))
    ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch in ax.artists:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor("#3b82f6")
        patch.set_alpha(0.7)
    rng = np.random.RandomState(13)
    for idx, vals in enumerate(data, start=1):
        if not vals:
            continue
        x = rng.normal(loc=idx, scale=0.035, size=len(vals))
        ax.scatter(x, vals, s=11, alpha=0.55, color="#1d4ed8")

    ax.set_title(f"Rank stability: {factor_name}", fontsize=10)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_xlabel("Factor level")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    fig.tight_layout()

    filename = f"{_safe_slug(factor_name)}_rank_stability.png"
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run_stability_analysis(
    db_path: str,
    analysis_id: int,
    config: dict[str, Any],
    include_robust: bool,
    artifact_dir: str = "artifacts",
) -> dict[str, Any]:
    runs = list_model_runs(db_path, analysis_id)
    if include_robust:
        selected = [r for r in runs if r["validity_class"] in {"valid", "valid_with_robust_se"}]
    else:
        selected = [r for r in runs if r["validity_class"] == "valid"]
    thresholds = {
        "engineering_delta": float(config.get("engineering_delta", 0.1)),
        "equivalence_bound": float(config.get("equivalence_bound", 0.05)),
        "stable_sign_pct": float(config.get("stable_sign_pct", 0.9)),
        "stable_practical_pct": float(config.get("stable_practical_pct", 0.8)),
        "non_effect_pct": float(config.get("non_effect_pct", 0.8)),
        "sign_flip_redflag_pct": float(config.get("sign_flip_redflag_pct", 0.6)),
    }
    continuous_covariates = _as_list(config.get("continuous_covariates"))
    categorical_covariates = _as_list(config.get("categorical_covariates"))
    primary_factors = _as_list(config.get("primary_factors"))
    interaction_candidates = _as_list(config.get("interaction_candidates"))
    plot_dir = _stability_plot_dir(artifact_dir, analysis_id)

    effects: list[dict[str, Any]] = []
    pair_map: dict[str, dict[str, Any]] = {}
    rank_map: dict[str, dict[str, Any]] = {}
    slope_map: dict[str, dict[str, Any]] = {}
    interaction_map: dict[str, dict[str, Any]] = {}

    for run in selected:
        model_class = run["model_class"]
        included_terms = run["included_terms"]
        coeffs = run["coeffs"]
        pvalues = run["pvalues"]
        for cov in continuous_covariates:
            coef_key = quote_name(cov)
            if coef_key in coeffs:
                key = f"SLOPE:{cov}"
                slot = slope_map.setdefault(key, {"values": [], "pvalues": [], "class_values": {}, "class_pvalues": {}, "model_values": []})
                val = float(coeffs[coef_key])
                slot["values"].append(val)
                slot["pvalues"].append(float(pvalues.get(coef_key, 1.0)))
                slot["class_values"].setdefault(model_class, []).append(val)
                slot["class_pvalues"].setdefault(model_class, []).append(float(pvalues.get(coef_key, 1.0)))
                slot["model_values"].append((val, included_terms))

        lsmeans = run["lsmeans"]
        for factor in primary_factors:
            fac = lsmeans.get(factor, {})
            for pair in fac.get("pairwise_differences", []):
                a = pair["level_a"]
                b = pair["level_b"]
                key = f"PAIRWISE_DIFF:{factor}:{a}:{b}"
                slot = pair_map.setdefault(key, {"factor": factor, "a": a, "b": b, "values": [], "class_values": {}, "model_values": []})
                val = float(pair["difference"])
                slot["values"].append(val)
                slot["class_values"].setdefault(model_class, []).append(val)
                slot["model_values"].append((val, included_terms))
            means = fac.get("adjusted_means", [])
            for rank, entry in enumerate(means, start=1):
                level = entry["level"]
                key = f"RANKING:{factor}:{level}"
                slot = rank_map.setdefault(key, {"factor": factor, "level": level, "values": [], "class_values": {}})
                slot["values"].append(float(rank))
                slot["class_values"].setdefault(model_class, []).append(float(rank))

        for inter in interaction_candidates:
            left_right = [p.strip() for p in inter.split(":")]
            if len(left_right) != 2:
                continue
            candidates = [k for k in coeffs if ":" in k and all(quote_name(x) in k or x in k for x in left_right)]
            if not candidates:
                continue
            value = max((abs(float(coeffs[k])) for k in candidates), default=0.0)
            key = f"INTERACTION_FLAG:{inter}"
            slot = interaction_map.setdefault(key, {"interaction": inter, "values": [], "class_values": {}, "model_values": []})
            slot["values"].append(value)
            slot["class_values"].setdefault(model_class, []).append(value)
            slot["model_values"].append((value, included_terms))

    candidate_covs = list(dict.fromkeys(categorical_covariates + continuous_covariates))
    rank_plot_paths: dict[str, str | None] = {}
    rank_by_factor: dict[str, dict[str, list[float]]] = {}
    for slot in rank_map.values():
        rank_by_factor.setdefault(str(slot["factor"]), {})[str(slot["level"])] = list(slot["values"])
    for factor, level_map in rank_by_factor.items():
        rank_plot_paths[factor] = _save_rank_stability_plot(plot_dir, factor, level_map)

    for key, slot in pair_map.items():
        sens = _effect_sensitivity(slot["model_values"], candidate_covs, thresholds["engineering_delta"])
        bucket, sign_consistency, practical_rate, equivalence_rate = _bucket_effect(
            slot["values"],
            thresholds["engineering_delta"],
            thresholds["equivalence_bound"],
            thresholds["stable_sign_pct"],
            thresholds["stable_practical_pct"],
            thresholds["non_effect_pct"],
            thresholds["sign_flip_redflag_pct"],
            has_sensitivity=bool(sens),
        )
        plot_path = _save_effect_distribution_plot(plot_dir, key, slot["values"], "Effect value (pairwise difference)")
        effects.append(
            {
                "effect_id": key,
                "effect_type": "PAIRWISE_DIFF",
                "factor_name": slot["factor"],
                "level_a": slot["a"],
                "level_b": slot["b"],
                "estimate_distribution": _estimate_distribution(slot["values"]),
                "sign_consistency": sign_consistency,
                "practical_rate": practical_rate,
                "equivalence_rate": equivalence_rate,
                "p_sig_rate": None,
                "sensitivity_flags": sens,
                "bucket": bucket,
                "plot_path": plot_path,
                "narrative": f'{slot["factor"]}: {slot["a"]} vs {slot["b"]} is {bucket.lower()} across valid models.',
                "stratified_summaries": _stratified_metrics(
                    slot["class_values"],
                    {},
                    thresholds,
                ),
            }
        )

    for key, slot in slope_map.items():
        sens = _effect_sensitivity(slot["model_values"], candidate_covs, thresholds["engineering_delta"])
        bucket, sign_consistency, practical_rate, equivalence_rate = _bucket_effect(
            slot["values"],
            thresholds["engineering_delta"],
            thresholds["equivalence_bound"],
            thresholds["stable_sign_pct"],
            thresholds["stable_practical_pct"],
            thresholds["non_effect_pct"],
            thresholds["sign_flip_redflag_pct"],
            has_sensitivity=bool(sens),
        )
        p_sig_rate = float(np.mean(np.array(slot["pvalues"]) < 0.05)) if slot["pvalues"] else None
        cov_name = key.split(":")[1]
        plot_path = _save_effect_distribution_plot(plot_dir, key, slot["values"], "Effect value (slope)")
        effects.append(
            {
                "effect_id": key,
                "effect_type": "SLOPE",
                "factor_name": cov_name,
                "level_a": None,
                "level_b": None,
                "estimate_distribution": _estimate_distribution(slot["values"]),
                "sign_consistency": sign_consistency,
                "practical_rate": practical_rate,
                "equivalence_rate": equivalence_rate,
                "p_sig_rate": p_sig_rate,
                "sensitivity_flags": sens,
                "bucket": bucket,
                "plot_path": plot_path,
                "narrative": f"Slope for {cov_name} is {bucket.lower()} across model universe.",
                "stratified_summaries": _stratified_metrics(
                    slot["class_values"],
                    slot["class_pvalues"],
                    thresholds,
                ),
            }
        )

    for key, slot in rank_map.items():
        values = slot["values"]
        dist = _estimate_distribution(values)
        iqr = dist["iqr"]
        bucket = "STABLE" if iqr <= 1.0 else "CONDITIONAL"
        effects.append(
            {
                "effect_id": key,
                "effect_type": "RANKING",
                "factor_name": slot["factor"],
                "level_a": slot["level"],
                "level_b": None,
                "estimate_distribution": dist,
                "sign_consistency": None,
                "practical_rate": None,
                "equivalence_rate": None,
                "p_sig_rate": None,
                "sensitivity_flags": {},
                "bucket": bucket,
                "plot_path": rank_plot_paths.get(str(slot["factor"])),
                "narrative": f'Level {slot["level"]} ranking for {slot["factor"]} has rank IQR {iqr:.2f}.',
                "stratified_summaries": {
                    cls: {
                        "estimate_distribution": _estimate_distribution(vals),
                        "n_models": len(vals),
                    }
                    for cls, vals in slot["class_values"].items()
                },
            }
        )

    for key, slot in interaction_map.items():
        sens = _effect_sensitivity(slot["model_values"], candidate_covs, thresholds["engineering_delta"])
        bucket, sign_consistency, practical_rate, equivalence_rate = _bucket_effect(
            slot["values"],
            thresholds["engineering_delta"],
            thresholds["equivalence_bound"],
            thresholds["stable_sign_pct"],
            thresholds["stable_practical_pct"],
            thresholds["non_effect_pct"],
            thresholds["sign_flip_redflag_pct"],
            has_sensitivity=bool(sens),
        )
        effects.append(
            {
                "effect_id": key,
                "effect_type": "INTERACTION_FLAG",
                "factor_name": slot["interaction"],
                "level_a": None,
                "level_b": None,
                "estimate_distribution": _estimate_distribution(slot["values"]),
                "sign_consistency": sign_consistency,
                "practical_rate": practical_rate,
                "equivalence_rate": equivalence_rate,
                "p_sig_rate": None,
                "sensitivity_flags": sens,
                "bucket": "CONDITIONAL" if bucket == "STABLE" else bucket,
                "plot_path": None,
                "narrative": f'Interaction {slot["interaction"]} signals conditional behavior.',
                "stratified_summaries": _stratified_metrics(slot["class_values"], {}, thresholds),
            }
        )

    effects.sort(key=lambda x: (x["bucket"], x["effect_type"], x["effect_id"]))
    summary = {
        "n_models_used": len(selected),
        "include_robust": include_robust,
        "bucket_counts": {
            "STABLE": sum(1 for e in effects if e["bucket"] == "STABLE"),
            "CONDITIONAL": sum(1 for e in effects if e["bucket"] == "CONDITIONAL"),
            "NON_EFFECT": sum(1 for e in effects if e["bucket"] == "NON_EFFECT"),
            "REDFLAG": sum(1 for e in effects if e["bucket"] == "REDFLAG"),
        },
    }
    stratified = {
        "ANOVA": {
            "n_models": sum(1 for r in selected if r["model_class"] == "ANOVA"),
        },
        "ANCOVA": {
            "n_models": sum(1 for r in selected if r["model_class"] == "ANCOVA"),
        },
    }

    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stability_results(
                analysis_id, created_at, include_robust, thresholds_json, summary_json, effects_json, stratified_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                analysis_id,
                utcnow_iso(),
                1 if include_robust else 0,
                to_json(thresholds),
                to_json(summary),
                to_json(effects),
                to_json(stratified),
            ],
        )
        conn.execute("UPDATE analyses SET status = 'stability_complete', updated_at = ? WHERE id = ?", [utcnow_iso(), analysis_id])
        conn.commit()
    return {"summary": summary, "effects": effects, "stratified": stratified, "thresholds": thresholds}


def latest_stability(db_path: str, analysis_id: int) -> dict[str, Any] | None:
    row = fetchone(
        db_path,
        """
        SELECT * FROM stability_results
        WHERE analysis_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        [analysis_id],
    )
    if not row:
        return None
    rec = dict(row)
    rec["thresholds"] = from_json(rec["thresholds_json"], {})
    rec["summary"] = from_json(rec["summary_json"], {})
    rec["effects"] = from_json(rec["effects_json"], [])
    rec["stratified"] = from_json(rec["stratified_json"], {})
    return rec
