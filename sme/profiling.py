from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .db import (
    fetchall,
    fetchone,
    from_json,
    get_conn,
    insert_and_get_id,
    to_json,
    utcnow_iso,
)
from .defaults import DEFAULT_PROFILE_RULES


@dataclass
class ProfileResult:
    schema: list[dict[str, Any]]
    columns: list[dict[str, Any]]


def allowed_roles_for_inferred_type(inferred_type: str) -> list[str]:
    if inferred_type in {"numeric", "datetime"}:
        return ["categorical", "continuous", "excluded"]
    return ["categorical", "excluded"]


def merged_rules(custom_rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = dict(DEFAULT_PROFILE_RULES)
    rules.update(custom_rules or {})
    return rules


def _sample_values(series: pd.Series, limit: int = 5) -> list[str]:
    values = series.dropna().head(limit).tolist()
    return [str(v) for v in values]


def _parse_ratio(parsed: pd.Series, original: pd.Series) -> float:
    non_null = original.notna().sum()
    if non_null == 0:
        return 0.0
    return float(parsed.notna().sum() / non_null)


def infer_profiles(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> ProfileResult:
    rules = merged_rules(rules)
    schema: list[dict[str, Any]] = []
    columns: list[dict[str, Any]] = []
    cat_max = int(rules["categorical_max_cardinality"])
    dt_cat_max = int(rules["datetime_categorical_max_cardinality"])
    txt_unique_cut = float(rules["text_unique_ratio_exclude"])
    num_ratio_cut = float(rules["numeric_parse_ratio"])
    dt_ratio_cut = float(rules["datetime_parse_ratio"])

    for col in df.columns:
        if col == "row_index":
            continue
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        numeric = pd.to_numeric(series, errors="coerce")
        datelike = pd.to_datetime(series, errors="coerce", utc=True)
        numeric_ratio = _parse_ratio(numeric, series)
        datetime_ratio = _parse_ratio(datelike, series)
        sample_values = _sample_values(series)
        inferred_type = "text"
        model_role = "excluded"
        exclude_reason = ""

        if numeric_ratio >= num_ratio_cut:
            inferred_type = "numeric"
            model_role = "categorical" if unique_count <= cat_max else "continuous"
        elif datetime_ratio >= dt_ratio_cut:
            inferred_type = "datetime"
            if unique_count <= dt_cat_max:
                model_role = "categorical"
            else:
                model_role = "continuous"
        else:
            inferred_type = "text"
            unique_ratio = float(unique_count / non_null) if non_null else 0.0
            if unique_count <= cat_max:
                model_role = "categorical"
            elif unique_ratio >= txt_unique_cut:
                model_role = "excluded"
                exclude_reason = "High-uniqueness text (likely ID/free text); not fit for model."
            else:
                model_role = "excluded"
                exclude_reason = "Text field exceeds categorical cardinality limit."

        schema.append({"column_name": col, "dtype": str(series.dtype), "inferred_type": inferred_type})
        columns.append(
            {
                "column_name": col,
                "inferred_type": inferred_type,
                "model_role": model_role,
                "unique_count": unique_count,
                "missing_count": missing,
                "numeric_ratio": numeric_ratio,
                "datetime_ratio": datetime_ratio,
                "exclude_reason": exclude_reason,
                "sample_values_json": json.dumps(sample_values, ensure_ascii=True),
            }
        )
    return ProfileResult(schema=schema, columns=columns)


def ingest_dataset(
    db_path: str,
    csv_path: str,
    dataset_name: str,
    original_filename: str,
    rules: dict[str, Any] | None = None,
) -> int:
    df = pd.read_csv(csv_path, low_memory=False)
    df.insert(0, "row_index", np.arange(len(df), dtype=int))
    profile = infer_profiles(df, rules)
    rules = merged_rules(rules)

    dataset_id = insert_and_get_id(
        db_path,
        """
        INSERT INTO datasets(
            name, original_filename, data_table_name, uploaded_at, updated_at, row_count, col_count, schema_json, profile_rules_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            dataset_name,
            original_filename,
            "pending",
            utcnow_iso(),
            utcnow_iso(),
            int(df.shape[0]),
            int(df.shape[1] - 1),
            to_json(profile.schema),
            to_json(rules),
        ],
    )
    table_name = f"dataset_data_{dataset_id}"
    with get_conn(db_path) as conn:
        conn.execute("UPDATE datasets SET data_table_name = ? WHERE id = ?", [table_name, dataset_id])
        conn.execute("DELETE FROM dataset_columns WHERE dataset_id = ?", [dataset_id])
        conn.executemany(
            """
            INSERT INTO dataset_columns(
                dataset_id, column_name, inferred_type, model_role, unique_count, missing_count,
                numeric_ratio, datetime_ratio, exclude_reason, sample_values_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    dataset_id,
                    col["column_name"],
                    col["inferred_type"],
                    col["model_role"],
                    col["unique_count"],
                    col["missing_count"],
                    col["numeric_ratio"],
                    col["datetime_ratio"],
                    col["exclude_reason"],
                    col["sample_values_json"],
                ]
                for col in profile.columns
            ],
        )
        conn.commit()
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    return dataset_id


def reprofile_dataset(db_path: str, dataset_id: int, rules: dict[str, Any]) -> None:
    df = load_dataset_dataframe(db_path, dataset_id)
    profile = infer_profiles(df, rules)
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE datasets SET schema_json = ?, profile_rules_json = ?, updated_at = ? WHERE id = ?",
            [to_json(profile.schema), to_json(rules), utcnow_iso(), dataset_id],
        )
        conn.execute("DELETE FROM dataset_columns WHERE dataset_id = ?", [dataset_id])
        conn.executemany(
            """
            INSERT INTO dataset_columns(
                dataset_id, column_name, inferred_type, model_role, unique_count, missing_count,
                numeric_ratio, datetime_ratio, exclude_reason, sample_values_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    dataset_id,
                    col["column_name"],
                    col["inferred_type"],
                    col["model_role"],
                    col["unique_count"],
                    col["missing_count"],
                    col["numeric_ratio"],
                    col["datetime_ratio"],
                    col["exclude_reason"],
                    col["sample_values_json"],
                ]
                for col in profile.columns
            ],
        )
        conn.commit()


def load_dataset_dataframe(db_path: str, dataset_id: int) -> pd.DataFrame:
    row = fetchone(db_path, "SELECT data_table_name FROM datasets WHERE id = ?", [dataset_id])
    if not row:
        raise ValueError(f"Unknown dataset_id={dataset_id}")
    table = row["data_table_name"]
    with get_conn(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def load_dataset_metadata(db_path: str, dataset_id: int) -> dict[str, Any]:
    dataset = fetchone(db_path, "SELECT * FROM datasets WHERE id = ?", [dataset_id])
    if not dataset:
        raise ValueError(f"Unknown dataset_id={dataset_id}")
    columns = fetchall(
        db_path,
        "SELECT * FROM dataset_columns WHERE dataset_id = ? ORDER BY column_name",
        [dataset_id],
    )
    return {
        "dataset": dict(dataset),
        "columns": [dict(c) for c in columns],
    }


def update_dataset_model_roles(db_path: str, dataset_id: int, role_updates: dict[int, str]) -> int:
    columns = fetchall(
        db_path,
        "SELECT id, inferred_type, model_role FROM dataset_columns WHERE dataset_id = ?",
        [dataset_id],
    )
    by_id = {int(c["id"]): dict(c) for c in columns}
    applied = 0
    with get_conn(db_path) as conn:
        for col_id, desired_role in role_updates.items():
            current = by_id.get(int(col_id))
            if not current:
                continue
            allowed = allowed_roles_for_inferred_type(str(current["inferred_type"]))
            next_role = desired_role if desired_role in allowed else str(current["model_role"])
            if next_role != str(current["model_role"]):
                exclude_reason = ""
                if next_role == "excluded":
                    exclude_reason = "User override: excluded from modeling."
                conn.execute(
                    """
                    UPDATE dataset_columns
                    SET model_role = ?, exclude_reason = ?
                    WHERE id = ? AND dataset_id = ?
                    """,
                    [next_role, exclude_reason, int(col_id), dataset_id],
                )
                applied += 1
        if applied > 0:
            conn.execute("UPDATE datasets SET updated_at = ? WHERE id = ?", [utcnow_iso(), dataset_id])
        conn.commit()
    return applied


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def dataset_column_value_preview(
    db_path: str,
    dataset_id: int,
    column_name: str,
    max_distinct_rows: int = 250,
    sample_rows: int = 25,
) -> dict[str, Any]:
    meta = load_dataset_metadata(db_path, dataset_id)
    dataset = meta["dataset"]
    columns = meta["columns"]
    names = {c["column_name"] for c in columns}
    if column_name not in names:
        raise ValueError(f"Unknown column: {column_name}")

    table = str(dataset["data_table_name"])
    quoted_table = _quote_identifier(table)
    quoted_col = _quote_identifier(column_name)
    distinct_limit = max(1, int(max_distinct_rows))
    sample_limit = max(1, int(sample_rows))

    with get_conn(db_path) as conn:
        distinct_rows = conn.execute(
            f"""
            SELECT {quoted_col} AS value, COUNT(*) AS n
            FROM {quoted_table}
            WHERE {quoted_col} IS NOT NULL
            GROUP BY {quoted_col}
            ORDER BY n DESC, value
            LIMIT ?
            """,
            [distinct_limit],
        ).fetchall()
        sample = conn.execute(
            f"""
            SELECT row_index, {quoted_col} AS value
            FROM {quoted_table}
            WHERE {quoted_col} IS NOT NULL
            ORDER BY row_index
            LIMIT ?
            """,
            [sample_limit],
        ).fetchall()
        non_null_count = conn.execute(
            f"SELECT COUNT(*) AS n FROM {quoted_table} WHERE {quoted_col} IS NOT NULL"
        ).fetchone()

    col_meta = next(c for c in columns if c["column_name"] == column_name)
    return {
        "column": col_meta,
        "non_null_count": int(non_null_count["n"]) if non_null_count else 0,
        "distinct_values": [{"value": r["value"], "count": int(r["n"])} for r in distinct_rows],
        "sample_rows": [{"row_index": int(r["row_index"]), "value": r["value"]} for r in sample],
    }


def _series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _group_label(row: pd.Series, group_vars: list[str]) -> str:
    if not group_vars:
        return "__ALL__"
    return "|".join(f"{g}={row[g]}" for g in group_vars)


def compute_univariate_screening(
    df: pd.DataFrame,
    columns_meta: list[dict[str, Any]],
    response: str,
) -> dict[str, Any]:
    y = _series_numeric(df, response)
    categorical = [c["column_name"] for c in columns_meta if c["model_role"] == "categorical" and c["column_name"] != response]
    continuous = [c["column_name"] for c in columns_meta if c["model_role"] == "continuous" and c["column_name"] != response]

    categorical_scores: list[dict[str, Any]] = []
    for col in categorical:
        temp = pd.DataFrame({"x": df[col], "y": y}).dropna()
        groups = [g["y"].to_numpy() for _, g in temp.groupby("x") if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        y_values = temp["y"].to_numpy()
        grand = np.mean(y_values)
        sst = np.sum((y_values - grand) ** 2)
        ssa = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        eta_sq = float(ssa / sst) if sst > 0 else 0.0
        categorical_scores.append(
            {
                "factor": col,
                "levels": int(temp["x"].nunique()),
                "n": int(len(temp)),
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "effect_size_eta_sq": eta_sq,
            }
        )

    continuous_scores: list[dict[str, Any]] = []
    for col in continuous:
        temp = pd.DataFrame({"x": _series_numeric(df, col), "y": y}).dropna()
        if len(temp) < 5:
            continue
        corr, corr_p = stats.pearsonr(temp["x"], temp["y"])
        lr = stats.linregress(temp["x"], temp["y"])
        continuous_scores.append(
            {
                "covariate": col,
                "n": int(len(temp)),
                "pearson_r": float(corr),
                "corr_p": float(corr_p),
                "slope": float(lr.slope),
                "slope_p": float(lr.pvalue),
            }
        )

    categorical_scores.sort(key=lambda x: x["effect_size_eta_sq"], reverse=True)
    continuous_scores.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    return {
        "categorical_univariate": categorical_scores,
        "continuous_univariate": continuous_scores,
        "likely_primary_factors": [x["factor"] for x in categorical_scores[:5]],
        "likely_important_covariates": [x["covariate"] for x in continuous_scores[:8]],
    }


def compute_confounding_map(
    df: pd.DataFrame,
    categorical_factors: list[str],
    covariates: list[str],
    columns_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    role_by_col = {c["column_name"]: c["model_role"] for c in columns_meta}
    records: list[dict[str, Any]] = []
    for factor in categorical_factors:
        if factor not in df.columns:
            continue
        for cov in covariates:
            if cov not in df.columns or cov == factor:
                continue
            if role_by_col.get(cov) == "continuous":
                temp = pd.DataFrame({"g": df[factor], "x": _series_numeric(df, cov)}).dropna()
                groups = [g["x"].to_numpy() for _, g in temp.groupby("g") if len(g) > 1]
                if len(groups) < 2:
                    continue
                stat, p_val = stats.f_oneway(*groups)
                method = "anova"
            else:
                temp = pd.DataFrame({"g": df[factor], "x": df[cov]}).dropna()
                if temp["g"].nunique() < 2 or temp["x"].nunique() < 2:
                    continue
                contingency = pd.crosstab(temp["g"], temp["x"])
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                stat = chi2
                method = "chi2"
            records.append(
                {
                    "factor": factor,
                    "covariate": cov,
                    "method": method,
                    "stat": float(stat),
                    "p_value": float(p_val),
                    "confounded": bool(p_val < 0.05),
                }
            )
    suspicious = [f'{r["factor"]} <- {r["covariate"]}' for r in records if r["confounded"]]
    return {"rows": records, "suspicious_confounders": suspicious[:30]}


def compute_collinearity_clusters(df: pd.DataFrame, covariates: list[str]) -> dict[str, Any]:
    numeric_cols = []
    for col in covariates:
        if col in df.columns:
            numeric_cols.append(col)
    if len(numeric_cols) < 2:
        return {"correlations": [], "clusters": []}

    matrix = pd.DataFrame({col: _series_numeric(df, col) for col in numeric_cols}).corr()
    edges = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1 :]:
            corr = matrix.loc[left, right]
            if pd.isna(corr):
                continue
            edges.append({"a": left, "b": right, "corr": float(corr)})

    strong_edges = [e for e in edges if abs(e["corr"]) >= 0.8]
    parent = {c: c for c in numeric_cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in strong_edges:
        union(e["a"], e["b"])
    groups: dict[str, list[str]] = {}
    for col in numeric_cols:
        groups.setdefault(find(col), []).append(col)
    clusters = [sorted(vals) for vals in groups.values() if len(vals) > 1]
    return {"correlations": strong_edges, "clusters": clusters}


def compute_health_checks(
    df: pd.DataFrame,
    response: str,
    group_variables: list[str],
    min_rows: int,
    min_rows_per_group: int,
) -> dict[str, Any]:
    rows = int(len(df))
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "minimum_rows",
            "passed": rows >= min_rows,
            "detail": f"Rows={rows}, required>={min_rows}",
        }
    )
    if response not in df.columns:
        checks.append({"name": "response_exists", "passed": False, "detail": f"Missing response column: {response}"})
    else:
        y = _series_numeric(df, response)
        valid = int(y.notna().sum())
        checks.append({"name": "response_numeric", "passed": valid >= min_rows, "detail": f"Numeric response rows={valid}"})

    if group_variables:
        temp = df[group_variables].fillna("__NA__")
        group_sizes = temp.groupby(group_variables).size().reset_index(name="count")
        min_group = int(group_sizes["count"].min()) if len(group_sizes) else 0
        checks.append(
            {
                "name": "group_size",
                "passed": min_group >= min_rows_per_group,
                "detail": f"Min group size={min_group}, required>={min_rows_per_group}",
            }
        )
    return {
        "rows": rows,
        "group_variables": group_variables,
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


def detect_group_outliers(
    df: pd.DataFrame,
    response: str,
    group_variables: list[str],
    threshold: float = 3.5,
) -> list[dict[str, Any]]:
    if response not in df.columns:
        return []
    y = _series_numeric(df, response)
    work = df.copy()
    work["_response"] = y
    work = work.dropna(subset=["_response"])
    if work.empty:
        return []

    groups = [("__ALL__", work)] if not group_variables else list(work.groupby(group_variables))
    flagged: list[dict[str, Any]] = []
    for key, grp in groups:
        values = grp["_response"]
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad <= 0:
            continue
        rz = 0.6745 * (values - med) / mad
        mask = np.abs(rz) > threshold
        if not mask.any():
            continue
        label = "__ALL__" if key == "__ALL__" else str(key)
        for row_idx, z in zip(grp.loc[mask, "row_index"], rz[mask]):
            flagged.append(
                {
                    "row_index": int(row_idx),
                    "group_key": label,
                    "variable": response,
                    "robust_z": float(z),
                    "decision": "keep",
                    "reason": f"|robust_z|>{threshold} (MAD within group)",
                    "created_at": utcnow_iso(),
                }
            )
    flagged.sort(key=lambda x: abs(x["robust_z"]), reverse=True)
    return flagged


def persist_outliers(db_path: str, analysis_id: int, outliers: list[dict[str, Any]]) -> None:
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM analysis_outliers WHERE analysis_id = ?", [analysis_id])
        conn.executemany(
            """
            INSERT INTO analysis_outliers(
                analysis_id, row_index, group_key, variable, robust_z, decision, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    analysis_id,
                    o["row_index"],
                    o["group_key"],
                    o["variable"],
                    o["robust_z"],
                    o["decision"],
                    o["reason"],
                    o["created_at"],
                ]
                for o in outliers
            ],
        )
        conn.commit()


def update_outlier_decisions(db_path: str, analysis_id: int, decisions: dict[int, str]) -> None:
    with get_conn(db_path) as conn:
        for row_index, decision in decisions.items():
            conn.execute(
                "UPDATE analysis_outliers SET decision = ? WHERE analysis_id = ? AND row_index = ?",
                [decision, analysis_id, row_index],
            )
        conn.commit()


def load_analysis_outliers(db_path: str, analysis_id: int) -> list[dict[str, Any]]:
    rows = fetchall(
        db_path,
        "SELECT * FROM analysis_outliers WHERE analysis_id = ? ORDER BY ABS(robust_z) DESC",
        [analysis_id],
    )
    return [dict(r) for r in rows]


def excluded_row_indices(db_path: str, analysis_id: int) -> set[int]:
    rows = fetchall(
        db_path,
        "SELECT row_index FROM analysis_outliers WHERE analysis_id = ? AND decision = 'remove'",
        [analysis_id],
    )
    return {int(r["row_index"]) for r in rows}


def build_screening_bundle(
    df: pd.DataFrame,
    columns_meta: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    response = config["response"]
    univariate = compute_univariate_screening(df, columns_meta, response)
    categorical_pool = [c["column_name"] for c in columns_meta if c["model_role"] == "categorical"]
    covariates = list(dict.fromkeys(config.get("categorical_covariates", []) + config.get("continuous_covariates", [])))
    confounding = compute_confounding_map(df, categorical_pool, covariates, columns_meta)
    collinear = compute_collinearity_clusters(df, config.get("continuous_covariates", []))
    return {"univariate": univariate, "confounding": confounding, "collinearity": collinear}
