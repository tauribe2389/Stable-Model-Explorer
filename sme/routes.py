from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from .db import fetchall, fetchone, from_json, get_conn, insert_and_get_id, to_json, utcnow_iso
from .defaults import DEFAULT_ANALYSIS_CONFIG, DEFAULT_PROFILE_RULES
from .modeling import (
    generate_model_registry,
    latest_stability,
    list_model_runs,
    load_registry,
    run_registry_models,
    run_stability_analysis,
)
from .profiling import (
    allowed_roles_for_inferred_type,
    build_screening_bundle,
    compute_health_checks,
    dataset_column_value_preview,
    detect_group_outliers,
    ingest_dataset,
    load_analysis_outliers,
    load_dataset_dataframe,
    load_dataset_metadata,
    merged_rules,
    persist_outliers,
    reprofile_dataset,
    update_dataset_model_roles,
    update_outlier_decisions,
)
from .reporting import create_report_snapshot

bp = Blueprint("sme", __name__)


def _db_path() -> str:
    return current_app.config["DATABASE"]


def _parse_int(name: str, default: int) -> int:
    try:
        return int(request.form.get(name, default))
    except Exception:
        return default


def _parse_float(name: str, default: float) -> float:
    try:
        return float(request.form.get(name, default))
    except Exception:
        return default


def _parse_bool(name: str, default: bool = False) -> bool:
    if name not in request.form:
        return default
    return request.form.get(name) in {"1", "true", "True", "on", "yes"}


def _parse_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _friendly_ts(ts: str | None) -> str:
    if not ts:
        return ""
    text = str(ts).strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        hour = dt.strftime("%I").lstrip("0") or "0"
        return f"{dt.strftime('%b')} {dt.day}, {dt.year} {hour}:{dt.strftime('%M %p')}"
    except Exception:
        return text


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _health_label(name: str) -> str:
    labels = {
        "minimum_rows": "Minimum row count",
        "response_exists": "Response column exists",
        "response_numeric": "Response is numeric",
        "group_size": "Minimum rows per group",
    }
    return labels.get(name, name.replace("_", " ").title())


def _build_config_from_form(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_ANALYSIS_CONFIG)
    if existing:
        cfg.update(existing)
    cfg.update(
        {
            "name": request.form.get("analysis_name", "").strip() or "SME Analysis",
            "response": request.form.get("response", "").strip(),
            "primary_factors": request.form.getlist("primary_factors"),
            "group_variables": request.form.getlist("group_variables"),
            "categorical_covariates": request.form.getlist("categorical_covariates"),
            "continuous_covariates": request.form.getlist("continuous_covariates"),
            "forced_terms": request.form.getlist("forced_terms"),
            "interaction_candidates": _parse_csv_list(request.form.get("interaction_candidates", "")),
            "max_covariates_in_model": _parse_int("max_covariates_in_model", cfg["max_covariates_in_model"]),
            "max_interactions_in_model": _parse_int("max_interactions_in_model", cfg["max_interactions_in_model"]),
            "max_total_models": _parse_int("max_total_models", cfg["max_total_models"]),
            "random_seed": _parse_int("random_seed", cfg["random_seed"]),
            "robust_se_mode": request.form.get("robust_se_mode", cfg["robust_se_mode"]).strip() or "HC3",
            "include_robust_in_stability": _parse_bool("include_robust_in_stability", True),
            "vif_fail_threshold": _parse_float("vif_fail_threshold", cfg["vif_fail_threshold"]),
            "vif_warn_threshold": _parse_float("vif_warn_threshold", cfg["vif_warn_threshold"]),
            "bp_alpha": _parse_float("bp_alpha", cfg["bp_alpha"]),
            "shapiro_fail_alpha": _parse_float("shapiro_fail_alpha", cfg["shapiro_fail_alpha"]),
            "shapiro_warn_alpha": _parse_float("shapiro_warn_alpha", cfg["shapiro_warn_alpha"]),
            "cook_threshold_multiplier": _parse_float("cook_threshold_multiplier", cfg["cook_threshold_multiplier"]),
            "cook_fail": _parse_bool("cook_fail", False),
            "outlier_mad_threshold": _parse_float("outlier_mad_threshold", cfg["outlier_mad_threshold"]),
            "stable_sign_pct": _parse_float("stable_sign_pct", cfg["stable_sign_pct"]),
            "stable_practical_pct": _parse_float("stable_practical_pct", cfg["stable_practical_pct"]),
            "non_effect_pct": _parse_float("non_effect_pct", cfg["non_effect_pct"]),
            "engineering_delta": _parse_float("engineering_delta", cfg["engineering_delta"]),
            "equivalence_bound": _parse_float("equivalence_bound", cfg["equivalence_bound"]),
            "sign_flip_redflag_pct": _parse_float("sign_flip_redflag_pct", cfg["sign_flip_redflag_pct"]),
            "health_min_rows": _parse_int("health_min_rows", cfg["health_min_rows"]),
            "health_min_rows_per_group": _parse_int("health_min_rows_per_group", cfg["health_min_rows_per_group"]),
        }
    )
    list_fields = [
        "primary_factors",
        "group_variables",
        "categorical_covariates",
        "continuous_covariates",
        "forced_terms",
        "interaction_candidates",
    ]
    for field in list_fields:
        cfg[field] = list(dict.fromkeys(cfg.get(field, [])))

    response = cfg.get("response", "")
    reserved = set(cfg.get("primary_factors", [])) | set(cfg.get("group_variables", [])) | ({response} if response else set())
    cfg["categorical_covariates"] = [c for c in cfg.get("categorical_covariates", []) if c not in reserved]
    cfg["continuous_covariates"] = [c for c in cfg.get("continuous_covariates", []) if c not in reserved]
    cfg["forced_terms"] = [c for c in cfg.get("forced_terms", []) if c != response]
    return cfg


@bp.route("/")
def home():
    db_path = _db_path()
    datasets = [dict(r) for r in fetchall(db_path, "SELECT * FROM datasets ORDER BY COALESCE(updated_at, uploaded_at) DESC")]
    analyses = [
        dict(r)
        for r in fetchall(
            db_path,
            """
            SELECT a.*, d.name AS dataset_name
            FROM analyses a
            JOIN datasets d ON d.id = a.dataset_id
            ORDER BY COALESCE(a.updated_at, a.created_at) DESC
            """,
        )
    ]
    for d in datasets:
        d["uploaded_at_friendly"] = _friendly_ts(d.get("uploaded_at"))
        d["updated_at_friendly"] = _friendly_ts(d.get("updated_at"))
    for a in analyses:
        a["created_at_friendly"] = _friendly_ts(a.get("created_at"))
        a["updated_at_friendly"] = _friendly_ts(a.get("updated_at"))
    presets = [dict(r) for r in fetchall(db_path, "SELECT * FROM analysis_presets ORDER BY name")]
    return render_template("home.html", datasets=datasets, analyses=analyses, presets=presets)


@bp.route("/upload", methods=["POST"])
def upload():
    db_path = _db_path()
    file = request.files.get("dataset_csv")
    if not file or not file.filename:
        flash("CSV file is required.", "error")
        return redirect(url_for("sme.home"))
    dataset_name = request.form.get("dataset_name", "").strip() or Path(file.filename).stem
    existing_dataset = fetchone(db_path, "SELECT id FROM datasets WHERE LOWER(name) = LOWER(?)", [dataset_name])
    if existing_dataset:
        flash(f'Dataset name "{dataset_name}" already exists. Use a unique name.', "error")
        return redirect(url_for("sme.home"))
    upload_dir = Path(current_app.config["UPLOAD_DIR"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / file.filename
    file.save(saved_path)

    rules = merged_rules(
        {
            "categorical_max_cardinality": _parse_int("categorical_max_cardinality", DEFAULT_PROFILE_RULES["categorical_max_cardinality"]),
            "datetime_categorical_max_cardinality": _parse_int(
                "datetime_categorical_max_cardinality", DEFAULT_PROFILE_RULES["datetime_categorical_max_cardinality"]
            ),
            "text_unique_ratio_exclude": _parse_float("text_unique_ratio_exclude", DEFAULT_PROFILE_RULES["text_unique_ratio_exclude"]),
        }
    )
    try:
        dataset_id = ingest_dataset(db_path, str(saved_path), dataset_name, file.filename, rules)
    except Exception as exc:
        flash(f"Upload failed: {exc}", "error")
        return redirect(url_for("sme.home"))
    flash(f"Dataset ingested (id={dataset_id}).", "success")
    return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))


@bp.route("/dataset/<int:dataset_id>")
def dataset_detail(dataset_id: int):
    db_path = _db_path()
    meta = load_dataset_metadata(db_path, dataset_id)
    dataset = meta["dataset"]
    columns = meta["columns"]
    for c in columns:
        c["allowed_roles"] = allowed_roles_for_inferred_type(str(c["inferred_type"]))
    profile_summary = {
        "categorical_count": sum(1 for c in columns if c.get("model_role") == "categorical"),
        "continuous_count": sum(1 for c in columns if c.get("model_role") == "continuous"),
        "excluded_count": sum(1 for c in columns if c.get("model_role") == "excluded"),
        "missing_columns_count": sum(1 for c in columns if int(c.get("missing_count", 0)) > 0),
    }
    dataset["schema"] = from_json(dataset["schema_json"], [])
    dataset["profile_rules"] = from_json(dataset["profile_rules_json"], DEFAULT_PROFILE_RULES)
    dataset["uploaded_at_friendly"] = _friendly_ts(dataset.get("uploaded_at"))
    dataset["updated_at_friendly"] = _friendly_ts(dataset.get("updated_at"))
    return render_template("dataset_detail.html", dataset=dataset, columns=columns, profile_summary=profile_summary)


@bp.route("/dataset/<int:dataset_id>/reprofile", methods=["POST"])
def reprofile(dataset_id: int):
    db_path = _db_path()
    rules = merged_rules(
        {
            "categorical_max_cardinality": _parse_int("categorical_max_cardinality", DEFAULT_PROFILE_RULES["categorical_max_cardinality"]),
            "datetime_categorical_max_cardinality": _parse_int(
                "datetime_categorical_max_cardinality", DEFAULT_PROFILE_RULES["datetime_categorical_max_cardinality"]
            ),
            "text_unique_ratio_exclude": _parse_float("text_unique_ratio_exclude", DEFAULT_PROFILE_RULES["text_unique_ratio_exclude"]),
        }
    )
    try:
        reprofile_dataset(db_path, dataset_id, rules)
        flash("Dataset profile rules updated.", "success")
    except Exception as exc:
        flash(f"Reprofile failed: {exc}", "error")
    return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))


@bp.route("/dataset/<int:dataset_id>/roles", methods=["POST"])
def update_dataset_roles(dataset_id: int):
    db_path = _db_path()
    meta = load_dataset_metadata(db_path, dataset_id)
    columns = meta["columns"]
    role_updates: dict[int, str] = {}
    for c in columns:
        field = f'role_{c["id"]}'
        role_updates[int(c["id"])] = request.form.get(field, str(c["model_role"]))
    try:
        count = update_dataset_model_roles(db_path, dataset_id, role_updates)
        flash(f"Updated model roles for {count} columns.", "success")
    except Exception as exc:
        flash(f"Role update failed: {exc}", "error")
    return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))


@bp.route("/dataset/<int:dataset_id>/column-preview")
def dataset_column_preview(dataset_id: int):
    db_path = _db_path()
    column_name = request.args.get("column_name", "").strip()
    if not column_name:
        flash("Choose a column to preview.", "error")
        return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))
    try:
        max_distinct = int(request.args.get("max_distinct", "40"))
    except Exception:
        max_distinct = 40
    try:
        sample_rows = int(request.args.get("sample_rows", "20"))
    except Exception:
        sample_rows = 20
    try:
        meta = load_dataset_metadata(db_path, dataset_id)
        preview = dataset_column_value_preview(
            db_path,
            dataset_id,
            column_name,
            max_distinct_rows=max_distinct,
            sample_rows=sample_rows,
        )
        dataset = meta["dataset"]
        role = str(preview["column"].get("model_role", "excluded"))
        detected = str(preview["column"].get("inferred_type", "text"))
        unique_count = int(preview["column"].get("unique_count", 0))
        missing_count = int(preview["column"].get("missing_count", 0))
        row_count = int(dataset.get("row_count", 0))
        non_null = int(preview.get("non_null_count", 0))
        allowed_roles = allowed_roles_for_inferred_type(detected)
        profile_rules = from_json(dataset.get("profile_rules_json"), DEFAULT_PROFILE_RULES)
        cat_limit = int(profile_rules.get("categorical_max_cardinality", DEFAULT_PROFILE_RULES["categorical_max_cardinality"]))

        if role == "categorical":
            if unique_count <= cat_limit and missing_count == 0:
                confidence_level = "good"
                confidence_message = "Suitable categorical factor (low cardinality, no missing data)."
            elif unique_count <= cat_limit:
                confidence_level = "ok"
                confidence_message = "Likely categorical factor; check missing values before modeling."
            else:
                confidence_level = "warn"
                confidence_message = "Categorical role selected with high cardinality; verify this is intentional."
        elif role == "continuous":
            if non_null >= max(30, int(0.6 * row_count)):
                confidence_level = "good"
                confidence_message = "Suitable continuous covariate coverage for modeling."
            else:
                confidence_level = "ok"
                confidence_message = "Continuous role selected with limited usable rows; monitor model stability."
        else:
            confidence_level = "ok"
            confidence_message = "Excluded from modeling; useful as traceability/ID/reference field."

        max_count = max((int(x["count"]) for x in preview["distinct_values"]), default=1)
        for item in preview["distinct_values"]:
            count = int(item["count"])
            item["pct_non_null"] = (count / non_null * 100.0) if non_null else 0.0
            item["bar_pct"] = (count / max_count * 100.0) if max_count else 0.0
        dominant_pct = max((x["pct_non_null"] for x in preview["distinct_values"]), default=0.0)
        imbalance_warning = dominant_pct > 40.0 and len(preview["distinct_values"]) > 1

        show_more_limit = min(max(100, max_distinct * 3), 3000)
        show_less_limit = 40
        return render_template(
            "dataset_column_preview.html",
            dataset=dataset,
            column_name=column_name,
            preview=preview,
            max_distinct=max_distinct,
            sample_rows=sample_rows,
            allowed_roles=allowed_roles,
            confidence_level=confidence_level,
            confidence_message=confidence_message,
            imbalance_warning=imbalance_warning,
            dominant_pct=dominant_pct,
            show_more_limit=show_more_limit,
            show_less_limit=show_less_limit,
        )
    except Exception as exc:
        flash(f"Column preview failed: {exc}", "error")
        return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))


@bp.route("/dataset/<int:dataset_id>/column-preview/role", methods=["POST"])
def dataset_column_preview_update_role(dataset_id: int):
    db_path = _db_path()
    column_name = request.form.get("column_name", "").strip()
    if not column_name:
        flash("Missing column name.", "error")
        return redirect(url_for("sme.dataset_detail", dataset_id=dataset_id))
    col_id = request.form.get("column_id", "").strip()
    role = request.form.get("model_role", "").strip()
    try:
        col_id_int = int(col_id)
    except Exception:
        flash("Invalid column identifier.", "error")
        return redirect(url_for("sme.dataset_column_preview", dataset_id=dataset_id, column_name=column_name))
    try:
        count = update_dataset_model_roles(db_path, dataset_id, {col_id_int: role})
        if count > 0:
            flash("Model role updated.", "success")
        else:
            flash("No role change applied.", "success")
    except Exception as exc:
        flash(f"Role update failed: {exc}", "error")
    return redirect(url_for("sme.dataset_column_preview", dataset_id=dataset_id, column_name=column_name))


@bp.route("/dataset/<int:dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id: int):
    db_path = _db_path()
    row = fetchone(db_path, "SELECT id, name, data_table_name FROM datasets WHERE id = ?", [dataset_id])
    if not row:
        flash("Dataset not found.", "error")
        return redirect(url_for("sme.home"))
    dataset = dict(row)
    table_name = _quote_ident(str(dataset["data_table_name"]))
    try:
        with get_conn(db_path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute("DELETE FROM datasets WHERE id = ?", [dataset_id])
            conn.commit()
        flash(f'Dataset "{dataset["name"]}" deleted.', "success")
    except Exception as exc:
        flash(f"Dataset delete failed: {exc}", "error")
    return redirect(url_for("sme.home"))


@bp.route("/analysis/<int:analysis_id>/delete", methods=["POST"])
def delete_analysis(analysis_id: int):
    db_path = _db_path()
    row = fetchone(db_path, "SELECT id, name FROM analyses WHERE id = ?", [analysis_id])
    if not row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(row)
    try:
        with get_conn(db_path) as conn:
            conn.execute("DELETE FROM analyses WHERE id = ?", [analysis_id])
            conn.commit()
        flash(f'Analysis "{analysis["name"]}" deleted.', "success")
    except Exception as exc:
        flash(f"Analysis delete failed: {exc}", "error")
    return redirect(url_for("sme.home"))


@bp.route("/analysis/new/<int:dataset_id>", methods=["GET", "POST"])
def analysis_new(dataset_id: int):
    db_path = _db_path()
    meta = load_dataset_metadata(db_path, dataset_id)
    dataset = meta["dataset"]
    columns = meta["columns"]
    response_candidates = [c["column_name"] for c in columns if c["model_role"] in {"continuous", "categorical"}]
    categorical = [c["column_name"] for c in columns if c["model_role"] == "categorical"]
    continuous = [c["column_name"] for c in columns if c["model_role"] == "continuous"]

    selected_config = dict(DEFAULT_ANALYSIS_CONFIG)
    preset_id = request.args.get("preset_id")
    if preset_id:
        preset = fetchone(db_path, "SELECT * FROM analysis_presets WHERE id = ?", [preset_id])
        if preset:
            selected_config.update(from_json(preset["config_json"], {}))

    if request.method == "POST":
        selected_config = _build_config_from_form(selected_config)
        response = selected_config["response"]
        if response not in response_candidates:
            flash("Select a valid response variable.", "error")
            return render_template(
                "analysis_wizard.html",
                dataset=dataset,
                columns=columns,
                config=selected_config,
                response_candidates=response_candidates,
                categorical=categorical,
                continuous=continuous,
                presets=[dict(r) for r in fetchall(db_path, "SELECT * FROM analysis_presets ORDER BY name")],
                preview={},
            )

        if request.form.get("save_preset_name", "").strip():
            name = request.form.get("save_preset_name", "").strip()
            now = utcnow_iso()
            with get_conn(db_path) as conn:
                existing = conn.execute("SELECT id FROM analysis_presets WHERE name = ?", [name]).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE analysis_presets SET config_json = ?, updated_at = ? WHERE id = ?",
                        [to_json(selected_config), now, existing["id"]],
                    )
                else:
                    conn.execute(
                        "INSERT INTO analysis_presets(name, config_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        [name, to_json(selected_config), now, now],
                    )
                conn.commit()
            flash(f"Preset saved: {name}", "success")

        analysis_name = selected_config.get("name", "SME Analysis")
        existing_analysis = fetchone(db_path, "SELECT id FROM analyses WHERE LOWER(name) = LOWER(?)", [analysis_name])
        if existing_analysis:
            flash(f'Analysis name "{analysis_name}" already exists. Use a unique name.', "error")
            return render_template(
                "analysis_wizard.html",
                dataset=dataset,
                columns=columns,
                config=selected_config,
                response_candidates=response_candidates,
                categorical=categorical,
                continuous=continuous,
                presets=[dict(r) for r in fetchall(db_path, "SELECT * FROM analysis_presets ORDER BY name")],
                preview={},
            )

        df = load_dataset_dataframe(db_path, dataset_id)
        screening = build_screening_bundle(df, columns, selected_config)
        health = compute_health_checks(
            df,
            selected_config["response"],
            selected_config["group_variables"],
            min_rows=int(selected_config["health_min_rows"]),
            min_rows_per_group=int(selected_config["health_min_rows_per_group"]),
        )
        analysis_id = insert_and_get_id(
            db_path,
            """
            INSERT INTO analyses(
                dataset_id, name, created_at, updated_at, status, config_json, screening_json, health_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                dataset_id,
                analysis_name,
                utcnow_iso(),
                utcnow_iso(),
                "configured",
                to_json(selected_config),
                to_json(screening),
                to_json(health),
                "",
            ],
        )

        outliers = detect_group_outliers(
            df,
            selected_config["response"],
            selected_config["group_variables"],
            threshold=float(selected_config["outlier_mad_threshold"]),
        )
        persist_outliers(db_path, analysis_id, outliers)
        generate_model_registry(db_path, analysis_id, selected_config, columns)
        with get_conn(db_path) as conn:
            conn.execute("UPDATE analyses SET status = 'registry_generated', updated_at = ? WHERE id = ?", [utcnow_iso(), analysis_id])
            conn.commit()
        flash("Analysis configured. Review outliers before running models.", "success")
        return redirect(url_for("sme.outlier_review", analysis_id=analysis_id))

    preview = {}
    if response_candidates:
        preview["suggested_response"] = response_candidates[0]
    return render_template(
        "analysis_wizard.html",
        dataset=dataset,
        columns=columns,
        config=selected_config,
        response_candidates=response_candidates,
        categorical=categorical,
        continuous=continuous,
        presets=[dict(r) for r in fetchall(db_path, "SELECT * FROM analysis_presets ORDER BY name")],
        preview=preview,
    )


@bp.route("/analysis/<int:analysis_id>/outliers", methods=["GET", "POST"])
def outlier_review(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    if request.method == "POST":
        outliers = load_analysis_outliers(db_path, analysis_id)
        decisions = {}
        for o in outliers:
            key = f'decision_{o["row_index"]}'
            decision = request.form.get(key, "keep")
            if decision not in {"keep", "remove"}:
                decision = "keep"
            decisions[int(o["row_index"])] = decision
        update_outlier_decisions(db_path, analysis_id, decisions)
        with get_conn(db_path) as conn:
            conn.execute("UPDATE analyses SET updated_at = ? WHERE id = ?", [utcnow_iso(), analysis_id])
            conn.commit()
        action = request.form.get("action", "save")
        flash("Outlier decisions saved.", "success")
        if action == "run":
            return redirect(url_for("sme.run_models", analysis_id=analysis_id))
        return redirect(url_for("sme.outlier_review", analysis_id=analysis_id))

    outliers = load_analysis_outliers(db_path, analysis_id)
    return render_template("outlier_review.html", analysis=analysis, outliers=outliers)


@bp.route("/analysis/<int:analysis_id>/registry")
def model_registry(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    config = from_json(analysis["config_json"], {})
    registry = load_registry(db_path, analysis_id)
    return render_template("model_registry.html", analysis=analysis, config=config, registry=registry)


@bp.route("/analysis/<int:analysis_id>/run", methods=["GET", "POST"])
def run_models(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    config = from_json(analysis["config_json"], {})
    dataset_id = int(analysis["dataset_id"])
    columns = load_dataset_metadata(db_path, dataset_id)["columns"]

    if request.method == "POST":
        try:
            counts = run_registry_models(db_path, current_app.config["ARTIFACT_DIR"], analysis_id, config, columns)
            flash(
                f"Model runs complete. valid={counts['valid']}, invalid={counts['invalid']}, robust={counts['valid_with_robust_se']}",
                "success",
            )
        except Exception as exc:
            flash(f"Model runs failed: {exc}", "error")
        return redirect(url_for("sme.model_runs_summary", analysis_id=analysis_id))

    registry = load_registry(db_path, analysis_id)
    return render_template("run_models.html", analysis=analysis, config=config, registry_count=len(registry))


@bp.route("/analysis/<int:analysis_id>/runs")
def model_runs_summary(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    all_runs = list_model_runs(db_path, analysis_id)
    runs = list(all_runs)
    validity_filter = request.args.get("validity", "").strip()
    if validity_filter:
        runs = [r for r in runs if r["validity_class"] == validity_filter]
    counts = {
        "total": len(all_runs),
        "valid": sum(1 for r in all_runs if r["validity_class"] == "valid"),
        "invalid": sum(1 for r in all_runs if r["validity_class"] == "invalid"),
        "valid_with_robust_se": sum(1 for r in all_runs if r["validity_class"] == "valid_with_robust_se"),
    }
    return render_template("model_runs_summary.html", analysis=analysis, runs=runs, counts=counts, validity_filter=validity_filter)


@bp.route("/analysis/<int:analysis_id>/stability", methods=["GET", "POST"])
def stability_dashboard(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    config = from_json(analysis["config_json"], {})

    if request.method == "POST":
        include_robust = _parse_bool("include_robust", bool(config.get("include_robust_in_stability", True)))
        try:
            run_stability_analysis(
                db_path,
                analysis_id,
                config,
                include_robust=include_robust,
                artifact_dir=current_app.config["ARTIFACT_DIR"],
            )
            flash("Stability analysis complete.", "success")
        except Exception as exc:
            flash(f"Stability analysis failed: {exc}", "error")
        return redirect(url_for("sme.stability_dashboard", analysis_id=analysis_id))

    stability = latest_stability(db_path, analysis_id)
    return render_template("stability_dashboard.html", analysis=analysis, config=config, stability=stability)


@bp.route("/analysis/<int:analysis_id>/report", methods=["GET", "POST"])
def report_export(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)

    if request.method == "POST":
        try:
            snapshot = create_report_snapshot(db_path, current_app.config["ARTIFACT_DIR"], analysis_id)
            flash(f"Report generated: {snapshot['pdf_path']}", "success")
        except Exception as exc:
            flash(f"Report generation failed: {exc}", "error")
        return redirect(url_for("sme.report_export", analysis_id=analysis_id))

    snapshots = [dict(r) for r in fetchall(db_path, "SELECT * FROM report_snapshots WHERE analysis_id = ? ORDER BY id DESC", [analysis_id])]
    return render_template("report_export.html", analysis=analysis, snapshots=snapshots)


@bp.route("/analysis/<int:analysis_id>")
def analysis_overview(analysis_id: int):
    db_path = _db_path()
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        flash("Analysis not found.", "error")
        return redirect(url_for("sme.home"))
    analysis = dict(analysis_row)
    dataset_row = fetchone(db_path, "SELECT id, name FROM datasets WHERE id = ?", [analysis["dataset_id"]])
    dataset = dict(dataset_row) if dataset_row else {"id": analysis["dataset_id"], "name": f"Dataset {analysis['dataset_id']}"}
    config = from_json(analysis["config_json"], {})
    screening = from_json(analysis.get("screening_json"), {})
    health = from_json(analysis.get("health_json"), {})
    config_summary = {
        "response": str(config.get("response", "")),
        "primary_factors": _safe_list(config.get("primary_factors")),
        "group_variables": _safe_list(config.get("group_variables")),
        "categorical_covariates": _safe_list(config.get("categorical_covariates")),
        "continuous_covariates": _safe_list(config.get("continuous_covariates")),
        "forced_terms": _safe_list(config.get("forced_terms")),
        "interaction_candidates": _safe_list(config.get("interaction_candidates")),
        "model_caps": {
            "max_covariates_in_model": int(config.get("max_covariates_in_model", 0)),
            "max_interactions_in_model": int(config.get("max_interactions_in_model", 0)),
            "max_total_models": int(config.get("max_total_models", 0)),
            "random_seed": int(config.get("random_seed", 0)),
        },
        "robust_se_mode": str(config.get("robust_se_mode", "HC3")),
    }
    checks = health.get("checks", []) if isinstance(health, dict) else []
    health_flags = [
        {
            "label": _health_label(str(c.get("name", ""))),
            "passed": bool(c.get("passed")),
            "detail": str(c.get("detail", "")),
        }
        for c in checks
    ]
    health_summary = {
        "passed": sum(1 for c in health_flags if c["passed"]),
        "failed": sum(1 for c in health_flags if not c["passed"]),
        "all_passed": bool(health.get("all_passed", False)),
        "rows": int(health.get("rows", 0)) if isinstance(health, dict) else 0,
    }
    univariate = screening.get("univariate", {}) if isinstance(screening, dict) else {}
    confounding = screening.get("confounding", {}) if isinstance(screening, dict) else {}
    collinearity = screening.get("collinearity", {}) if isinstance(screening, dict) else {}
    screening_digest = {
        "likely_primary_factors": univariate.get("likely_primary_factors", []),
        "likely_important_covariates": univariate.get("likely_important_covariates", []),
        "categorical_univariate": univariate.get("categorical_univariate", [])[:12],
        "continuous_univariate": univariate.get("continuous_univariate", [])[:12],
        "suspicious_confounders": confounding.get("suspicious_confounders", [])[:20],
        "collinearity_clusters": collinearity.get("clusters", []),
    }
    return render_template(
        "analysis_overview.html",
        analysis=analysis,
        dataset=dataset,
        config_summary=config_summary,
        health_flags=health_flags,
        health_summary=health_summary,
        screening_digest=screening_digest,
    )


@bp.route("/download")
def download_file():
    raw_path = request.args.get("path", "")
    if not raw_path:
        flash("Missing file path.", "error")
        return redirect(url_for("sme.home"))
    p = Path(raw_path)
    try:
        p = p.resolve()
    except Exception:
        flash("Invalid path.", "error")
        return redirect(url_for("sme.home"))
    allowed_root = Path(current_app.config["ARTIFACT_DIR"]).resolve()
    if allowed_root not in p.parents and p != allowed_root:
        flash("Path not allowed.", "error")
        return redirect(url_for("sme.home"))
    if not p.exists():
        flash("File not found.", "error")
        return redirect(url_for("sme.home"))
    return send_file(p, as_attachment=True)


@bp.route("/artifact")
def view_artifact():
    raw_path = request.args.get("path", "")
    if not raw_path:
        flash("Missing file path.", "error")
        return redirect(url_for("sme.home"))
    p = Path(raw_path)
    try:
        p = p.resolve()
    except Exception:
        flash("Invalid path.", "error")
        return redirect(url_for("sme.home"))
    allowed_root = Path(current_app.config["ARTIFACT_DIR"]).resolve()
    if allowed_root not in p.parents and p != allowed_root:
        flash("Path not allowed.", "error")
        return redirect(url_for("sme.home"))
    if not p.exists():
        flash("Artifact not found.", "error")
        return redirect(url_for("sme.home"))
    return send_file(p, as_attachment=False)
