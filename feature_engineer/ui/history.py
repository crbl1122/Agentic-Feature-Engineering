"""
History rendering — HTML table for Gradio UI.
"""
import json

from feature_engineer.storage.database import init_db, load_failed_runs
from feature_engineer.config import DB_PATH
import sqlite3


def load_history_html(db_path: str = DB_PATH) -> str:
    """Render run history as an HTML table with stacked features."""
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        try:
            rows = con.execute(
                "SELECT thread_id, timestamp, input_file, hint, attempts, status, features_json "
                "FROM runs ORDER BY timestamp DESC"
            ).fetchall()
        except Exception:
            return "<p>No runs yet.</p>"

    if not rows:
        return "<p style='color:var(--color-text-secondary)'>No runs yet.</p>"

    status_color = {"success": "#0F6E56", "failed": "#993C1D"}

    html = """
    <style>
      .hist-table { width:100%; border-collapse:collapse; font-size:13px; }
      .hist-table th { text-align:left; padding:8px 10px;
                       border-bottom:2px solid var(--color-border-secondary);
                       color:var(--color-text-secondary); font-weight:500; white-space:nowrap; }
      .hist-table td { padding:8px 10px; border-bottom:1px solid var(--color-border-tertiary);
                       vertical-align:top; word-break:break-word; }
      .feat-row  { margin-bottom:6px; }
      .feat-name { font-weight:500; color:var(--color-text-primary); }
      .feat-desc { color:var(--color-text-secondary); font-size:12px; }
      .feat-code { font-family:var(--font-mono); font-size:11px;
                   color:var(--color-text-tertiary); word-break:break-all; }
      .badge     { display:inline-block; padding:2px 8px; border-radius:10px;
                   font-size:11px; font-weight:500; }
      .tid       { font-family:var(--font-mono); font-size:10px;
                   color:var(--color-text-tertiary); }
    </style>
    <table class="hist-table">
      <thead><tr>
        <th>Timestamp</th><th>File</th><th>Objective</th>
        <th>Features</th><th>Attempts</th><th>Status</th>
      </tr></thead><tbody>
    """

    for thread_id, timestamp, input_file, hint, attempts, status, features_json in rows:
        plans = json.loads(features_json) if features_json else []
        color = status_color.get(status, "#444")

        if plans:
            features_html = "".join(
                f"""<div class="feat-row">
                      <div class="feat-name">{p['feature_name']}</div>
                      <div class="feat-desc">{p['description']}</div>
                      <div class="feat-code">{p['pandas_code']}</div>
                    </div>"""
                for p in plans
            )
        else:
            features_html = "<span style='color:var(--color-text-tertiary)'>—</span>"

        html += f"""
        <tr>
          <td><span>{timestamp}</span><br>
              <span class="tid">{thread_id[:18]}…</span></td>
          <td>{input_file or '—'}</td>
          <td>{hint or '—'}</td>
          <td>{features_html}</td>
          <td style="text-align:center">{attempts}</td>
          <td><span class="badge"
                    style="background:{color}22;color:{color}">{status}</span></td>
        </tr>"""

    html += "</tbody></table>"
    return html
