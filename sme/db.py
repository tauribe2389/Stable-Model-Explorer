from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn(db_path: str):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def ensure_db(db_path: str) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                data_table_name TEXT NOT NULL UNIQUE,
                uploaded_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                col_count INTEGER NOT NULL,
                schema_json TEXT NOT NULL,
                profile_rules_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_columns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                inferred_type TEXT NOT NULL,
                model_role TEXT NOT NULL,
                unique_count INTEGER NOT NULL,
                missing_count INTEGER NOT NULL,
                numeric_ratio REAL NOT NULL,
                datetime_ratio REAL NOT NULL,
                exclude_reason TEXT,
                sample_values_json TEXT NOT NULL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_dataset_columns_dataset ON dataset_columns(dataset_id);

            CREATE TABLE IF NOT EXISTS analysis_presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                screening_json TEXT NOT NULL DEFAULT '{}',
                health_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT DEFAULT '',
                FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_dataset ON analyses(dataset_id);

            CREATE TABLE IF NOT EXISTS analysis_outliers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                group_key TEXT NOT NULL,
                variable TEXT NOT NULL,
                robust_z REAL NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(analysis_id, row_index, variable),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_outliers_analysis ON analysis_outliers(analysis_id);

            CREATE TABLE IF NOT EXISTS model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                model_idx INTEGER NOT NULL,
                formula TEXT NOT NULL,
                model_class TEXT NOT NULL,
                included_terms_json TEXT NOT NULL,
                interactions_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                UNIQUE(analysis_id, model_idx),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_model_registry_analysis ON model_registry(analysis_id);

            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                registry_id INTEGER NOT NULL,
                run_at TEXT NOT NULL,
                status TEXT NOT NULL,
                validity_class TEXT NOT NULL,
                invalid_reasons_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                coeffs_json TEXT NOT NULL,
                pvalues_json TEXT NOT NULL,
                ci_json TEXT NOT NULL,
                anova_json TEXT NOT NULL,
                lsmeans_json TEXT NOT NULL,
                cooks_json TEXT NOT NULL,
                residual_diag_json TEXT NOT NULL,
                artifacts_json TEXT NOT NULL,
                n_obs INTEGER NOT NULL,
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
                FOREIGN KEY(registry_id) REFERENCES model_registry(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_model_runs_analysis ON model_runs(analysis_id);
            CREATE INDEX IF NOT EXISTS idx_model_runs_registry ON model_runs(registry_id);

            CREATE TABLE IF NOT EXISTS stability_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                include_robust INTEGER NOT NULL,
                thresholds_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                effects_json TEXT NOT NULL,
                stratified_json TEXT NOT NULL,
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_stability_analysis ON stability_results(analysis_id);

            CREATE TABLE IF NOT EXISTS report_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                html_path TEXT NOT NULL,
                pdf_path TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_reports_analysis ON report_snapshots(analysis_id);
            """
        )
        dataset_cols = {r["name"] for r in conn.execute("PRAGMA table_info(datasets)").fetchall()}
        analyses_cols = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()}
        if "updated_at" not in dataset_cols:
            conn.execute("ALTER TABLE datasets ADD COLUMN updated_at TEXT")
        if "updated_at" not in analyses_cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN updated_at TEXT")

        now = utcnow_iso()
        conn.execute(
            "UPDATE datasets SET updated_at = COALESCE(updated_at, uploaded_at, ?) WHERE updated_at IS NULL OR updated_at = ''",
            [now],
        )
        conn.execute(
            "UPDATE analyses SET updated_at = COALESCE(updated_at, created_at, ?) WHERE updated_at IS NULL OR updated_at = ''",
            [now],
        )

        dataset_dups = conn.execute(
            "SELECT LOWER(name) AS lname, COUNT(*) AS n FROM datasets GROUP BY LOWER(name) HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        analyses_dups = conn.execute(
            "SELECT LOWER(name) AS lname, COUNT(*) AS n FROM analyses GROUP BY LOWER(name) HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if not dataset_dups:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_datasets_name_nocase ON datasets(name COLLATE NOCASE)")
        if not analyses_dups:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_analyses_name_nocase ON analyses(name COLLATE NOCASE)")
        conn.commit()


def execute(db_path: str, sql: str, params: Iterable[Any] | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(sql, tuple(params or ()))
        conn.commit()


def executemany(db_path: str, sql: str, rows: Iterable[Iterable[Any]]) -> None:
    with get_conn(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()


def insert_and_get_id(db_path: str, sql: str, params: Iterable[Any]) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return int(cur.lastrowid)


def fetchall(db_path: str, sql: str, params: Iterable[Any] | None = None) -> list[sqlite3.Row]:
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, tuple(params or ()))
        return list(cur.fetchall())


def fetchone(db_path: str, sql: str, params: Iterable[Any] | None = None) -> sqlite3.Row | None:
    with get_conn(db_path) as conn:
        cur = conn.execute(sql, tuple(params or ()))
        return cur.fetchone()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def from_json(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default
