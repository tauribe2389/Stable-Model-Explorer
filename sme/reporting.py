from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import fetchall, fetchone, from_json, get_conn, to_json, utcnow_iso
from .modeling import latest_stability, list_model_runs
from .profiling import load_analysis_outliers


def _run_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(runs),
        "valid": sum(1 for r in runs if r["validity_class"] == "valid"),
        "invalid": sum(1 for r in runs if r["validity_class"] == "invalid"),
        "valid_with_robust_se": sum(1 for r in runs if r["validity_class"] == "valid_with_robust_se"),
    }


def _html_escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report_html(
    analysis: dict[str, Any],
    dataset: dict[str, Any],
    runs: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
    stability: dict[str, Any] | None,
) -> str:
    cfg = from_json(analysis["config_json"], {})
    screening = from_json(analysis.get("screening_json"), {})
    health = from_json(analysis.get("health_json"), {})
    counts = _run_counts(runs)
    rows = [
        "<tr><th>Model</th><th>Class</th><th>Validity</th><th>R2</th><th>Max VIF</th><th>Warnings</th></tr>"
    ]
    for r in runs[:200]:
        metrics = r.get("metrics", {})
        rows.append(
            "<tr>"
            f"<td>{r['model_idx']}</td>"
            f"<td>{_html_escape(r['model_class'])}</td>"
            f"<td>{_html_escape(r['validity_class'])}</td>"
            f"<td>{metrics.get('r2', '')}</td>"
            f"<td>{metrics.get('max_vif', '')}</td>"
            f"<td>{_html_escape('; '.join(r.get('warnings', [])))}</td>"
            "</tr>"
        )
    stability_block = ""
    if stability:
        bucket_counts = stability["summary"].get("bucket_counts", {})
        effects_rows = []
        for eff in stability.get("effects", [])[:200]:
            effects_rows.append(
                "<tr>"
                f"<td>{_html_escape(eff['effect_id'])}</td>"
                f"<td>{_html_escape(eff['effect_type'])}</td>"
                f"<td>{_html_escape(eff['bucket'])}</td>"
                f"<td>{eff.get('sign_consistency', '')}</td>"
                f"<td>{eff.get('practical_rate', '')}</td>"
                "</tr>"
            )
        stability_block = f"""
        <h2>Stability Analysis</h2>
        <p>Models used: {stability['summary'].get('n_models_used', 0)} | Include robust: {stability['summary'].get('include_robust', False)}</p>
        <p>Bucket counts: STABLE={bucket_counts.get('STABLE', 0)}, CONDITIONAL={bucket_counts.get('CONDITIONAL', 0)},
        NON_EFFECT={bucket_counts.get('NON_EFFECT', 0)}, REDFLAG={bucket_counts.get('REDFLAG', 0)}</p>
        <table>
          <tr><th>Effect</th><th>Type</th><th>Bucket</th><th>Sign Consistency</th><th>Practical Rate</th></tr>
          {''.join(effects_rows)}
        </table>
        """

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SME Report - Analysis {analysis['id']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 12px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 12px; text-align: left; }}
    .mono {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>Stable Model Explorer Report</h1>
  <p>Analysis ID: <span class="mono">{analysis['id']}</span></p>
  <p>Dataset: <span class="mono">{_html_escape(dataset['name'])}</span> ({dataset['row_count']} rows, {dataset['col_count']} columns)</p>
  <p>Status: {_html_escape(analysis['status'])}</p>
  <h2>Configuration Snapshot</h2>
  <pre>{_html_escape(cfg)}</pre>
  <h2>Health Checks</h2>
  <pre>{_html_escape(health)}</pre>
  <h2>Screening Snapshot</h2>
  <pre>{_html_escape(screening)}</pre>
  <h2>Outlier Review</h2>
  <p>Flagged rows: {len(outliers)} | Removed rows: {sum(1 for o in outliers if o['decision'] == 'remove')}</p>
  <h2>Model Runs</h2>
  <p>Total={counts['total']}, Valid={counts['valid']}, Invalid={counts['invalid']}, Valid-with-robust={counts['valid_with_robust_se']}</p>
  <table>{''.join(rows)}</table>
  {stability_block}
</body>
</html>
"""
    return html


def build_report_pdf(
    pdf_path: str,
    analysis: dict[str, Any],
    dataset: dict[str, Any],
    runs: list[dict[str, Any]],
    stability: dict[str, Any] | None,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    counts = _run_counts(runs)

    story.append(Paragraph("Stable Model Explorer Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Analysis ID: {analysis['id']}", styles["Normal"]))
    story.append(Paragraph(f"Dataset: {dataset['name']} ({dataset['row_count']} rows)", styles["Normal"]))
    story.append(Paragraph(f"Status: {analysis['status']}", styles["Normal"]))
    story.append(Spacer(1, 8))

    run_table = Table(
        [
            ["Total Models", "Valid", "Invalid", "Valid+Robust"],
            [counts["total"], counts["valid"], counts["invalid"], counts["valid_with_robust_se"]],
        ]
    )
    run_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eefb")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]
        )
    )
    story.append(run_table)
    story.append(Spacer(1, 10))

    if stability:
        bucket = stability["summary"].get("bucket_counts", {})
        bucket_table = Table(
            [
                ["Stable", "Conditional", "Non-effect", "Red-flag"],
                [
                    bucket.get("STABLE", 0),
                    bucket.get("CONDITIONAL", 0),
                    bucket.get("NON_EFFECT", 0),
                    bucket.get("REDFLAG", 0),
                ],
            ]
        )
        bucket_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee8d6")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        story.append(Paragraph("Stability Buckets", styles["Heading2"]))
        story.append(bucket_table)
        story.append(Spacer(1, 8))

    story.append(Paragraph("Top Models", styles["Heading2"]))
    top = sorted(runs, key=lambda r: r.get("metrics", {}).get("adj_r2", -999), reverse=True)[:10]
    table_data = [["Model", "Class", "Validity", "Adj R2", "Max VIF"]]
    for r in top:
        m = r.get("metrics", {})
        table_data.append([r["model_idx"], r["model_class"], r["validity_class"], f"{m.get('adj_r2', '')}", f"{m.get('max_vif', '')}"])
    model_table = Table(table_data)
    model_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    story.append(model_table)
    doc.build(story)


def create_report_snapshot(db_path: str, artifact_dir: str, analysis_id: int) -> dict[str, Any]:
    analysis_row = fetchone(db_path, "SELECT * FROM analyses WHERE id = ?", [analysis_id])
    if not analysis_row:
        raise ValueError(f"Unknown analysis_id={analysis_id}")
    analysis = dict(analysis_row)
    dataset_row = fetchone(db_path, "SELECT * FROM datasets WHERE id = ?", [analysis["dataset_id"]])
    dataset = dict(dataset_row) if dataset_row else {"name": "unknown", "row_count": 0, "col_count": 0}
    runs = list_model_runs(db_path, analysis_id)
    outliers = load_analysis_outliers(db_path, analysis_id)
    stability = latest_stability(db_path, analysis_id)

    report_dir = Path(artifact_dir) / "reports" / f"analysis_{analysis_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utcnow_iso().replace(":", "-")
    html_path = report_dir / f"report_{timestamp}.html"
    pdf_path = report_dir / f"report_{timestamp}.pdf"

    html = build_report_html(analysis, dataset, runs, outliers, stability)
    html_path.write_text(html, encoding="utf-8")
    build_report_pdf(str(pdf_path), analysis, dataset, runs, stability)

    meta = {
        "counts": _run_counts(runs),
        "has_stability": bool(stability),
        "outliers": {
            "flagged": len(outliers),
            "removed": sum(1 for o in outliers if o["decision"] == "remove"),
        },
    }
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO report_snapshots(
                analysis_id, created_at, html_path, pdf_path, meta_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                analysis_id,
                utcnow_iso(),
                str(html_path),
                str(pdf_path),
                to_json(meta),
            ],
        )
        conn.commit()
    return {"html_path": str(html_path), "pdf_path": str(pdf_path), "meta": meta}
