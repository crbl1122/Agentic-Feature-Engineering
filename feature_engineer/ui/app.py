"""
Gradio UI — async run_agent, resume_run, launch_ui.
"""
import os
import sys
import traceback
import uuid

import pandas as pd
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from feature_engineer.config import DB_PATH, OUTPUT_DIR, DEFAULT_MAX_FEATURES, DEFAULT_RECURSION_LIMIT
from feature_engineer.graph.builder import build_graph
from feature_engineer.state import empty_state
from feature_engineer.storage.database import init_db, save_run, load_failed_runs
from feature_engineer.storage.parquet import init_dirs
from feature_engineer.ui.history import load_history_html


# ── Async streaming agent ───────────────────────────────────────────────────────

async def run_agent(csv_file, objective: str, max_features: int, exclude_columns: str = ""):
    """Async streaming generator — yields log updates as the agent runs."""
    import gradio as gr

    if csv_file is None:
        raise gr.Error("Please upload a CSV file first.")

    init_db()
    init_dirs()

    input_filename = os.path.splitext(os.path.basename(csv_file.name))[0]
    input_path     = csv_file.name
    output_path    = os.path.join(OUTPUT_DIR, f"{input_filename}_enriched.csv")
    thread_id      = str(uuid.uuid4())
    config         = {
        "configurable":    {"thread_id": thread_id},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }

    # parse exclude_columns — one per line or comma-separated
    target_cols = []
    if exclude_columns and exclude_columns.strip():
        for part in exclude_columns.replace(",", "\n").splitlines():
            col = part.strip().strip("-").strip()
            if col:
                target_cols.append(col)

    initial_state = empty_state(
        input_path=input_path,
        output_path=output_path,
        objective=objective.strip(),
        max_features=int(max_features),
        target_columns=target_cols,
    )

    logs        = ""
    status      = "success"
    final_state = dict(initial_state)

    class _Capture:
        def __init__(self): self.buf = ""
        def write(self, t): self.buf += t
        def flush(self):    pass
        def drain(self) -> str:
            out, self.buf = self.buf, ""
            return out

    capture    = _Capture()
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            graph = build_graph().compile(checkpointer=checkpointer)
            try:
                async for chunk in graph.astream(initial_state, config=config,
                                                  stream_mode="updates"):
                    for node_name in chunk:
                        new_output = capture.drain()
                        logs += new_output if new_output.strip() \
                            else f"[{node_name}] step complete\n"
                        yield logs, "", pd.DataFrame(), None, ""

                final_state = dict((await graph.aget_state(config)).values)

            except Exception:
                sys.stdout = old_stdout
                status      = "failed"
                logs       += capture.drain() + traceback.format_exc()
                yield logs, "Agent failed — check logs.", pd.DataFrame(), None, ""

    except Exception:
        status  = "failed"
        logs   += capture.drain() + traceback.format_exc()
        yield logs, "Checkpointer failed — check logs.", pd.DataFrame(), None, ""
    finally:
        sys.stdout = old_stdout

    completed       = final_state.get("completed_features", [])
    recommendations = final_state.get("feature_recommendations", [])
    df_out          = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()
    save_run(thread_id, final_state, status)

    summary = (
        f"Thread ID          : {thread_id}\n"
        f"Features completed : {len(completed)} → {completed}\n"
        f"Status             : {status}\n"
        f"Output shape       : {df_out.shape[0]} rows × {df_out.shape[1]} cols\n"
    )
    if recommendations:
        summary += "\n── Recommended features (require additional data) ──────────\n"
        for i, rec in enumerate(recommendations, 1):
            summary += (
                f"\n{i}. {rec.get('name', '')}\n"
                f"   {rec.get('description', '')}\n"
                f"   Needs: {rec.get('required_data', '')}\n"
            )

    yield logs, summary, df_out.head(20), output_path, load_history_html()


def _format_recommendations(recommendations: list[dict]) -> str:
    """Format recommendations as HTML for Gradio."""
    if not recommendations:
        return "<p style='color:gray'>No recommendations generated yet. Run the agent first.</p>"
    html = "<h3>Top 5 Recommended Features (require additional data)</h3>"
    for i, rec in enumerate(recommendations, 1):
        html += f"""
        <div style='border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0'>
            <b>{i}. {rec.get('name', '')}</b><br>
            <span style='color:#555'>{rec.get('description', '')}</span><br><br>
            <b>Required data:</b> {rec.get('required_data', '')}<br>
            <b>Example formula:</b> <code>{rec.get('example_formula', '')}</code>
        </div>"""
    return html


async def resume_run(thread_id: str):
    import gradio as gr

    if not thread_id or not thread_id.strip():
        raise gr.Error("Select a failed run first.")

    thread_id   = thread_id.strip()
    output_path = os.path.join(OUTPUT_DIR, "resumed_enriched.csv")
    config      = {
        "configurable":    {"thread_id": thread_id},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }

    logs        = ""
    status      = "success"
    final_state = {}

    try:
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
            graph = build_graph().compile(checkpointer=checkpointer)
            final_state = dict((await graph.ainvoke(None, config=config)).values
                               if hasattr(await graph.ainvoke(None, config=config), "values")
                               else await graph.ainvoke(None, config=config))
    except Exception:
        status      = "failed"
        logs       += traceback.format_exc()

    completed = final_state.get("completed_features", [])
    df_out    = pd.read_csv(output_path) if os.path.exists(output_path) else pd.DataFrame()
    save_run(thread_id, final_state, status)

    summary = (
        f"Thread ID          : {thread_id}\n"
        f"Features completed : {len(completed)} → {completed}\n"
        f"Status             : {status}\n"
        f"Output shape       : {df_out.shape[0]} rows × {df_out.shape[1]} cols"
    )
    return logs, summary, df_out.head(20), output_path, load_history_html()


# ── Gradio layout ───────────────────────────────────────────────────────────────

def launch_ui() -> None:
    import gradio as gr

    init_db()

    with gr.Blocks(title="LangGraph Feature Engineering Agent",
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
# LangGraph Feature Engineering Agent
Describe your ML objective, upload a CSV, and let the agent research,
plan, generate and validate features automatically.
        """)

        with gr.Tabs():

            # ── Run ────────────────────────────────────────────────────────────
            with gr.Tab("Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        csv_input  = gr.File(
                            label="Upload CSV",
                            file_types=[".csv"],
                            file_count="single",
                        )
                        clear_btn = gr.ClearButton(
                            components=[csv_input],
                            value="Clear",
                            size="sm",
                        )
                        hint_input = gr.Textbox(
                            label="Objective",
                            placeholder="e.g. predict sales using location OR time based features only",
                            lines=2,
                        )
                        exclude_input = gr.Textbox(
                            label="Exclude columns (one per line, leave empty to use all)",
                            placeholder="e.g.\nLN_IC50\nAUC\nZ_SCORE",
                            lines=3,
                        )
                        run_btn             = gr.Button("Run Agent", variant="primary")
                        max_features_slider = gr.Slider(
                            minimum=1, maximum=10, value=DEFAULT_MAX_FEATURES,
                            step=1, label="Max features to generate",
                        )
                    with gr.Column(scale=2):
                        summary_box  = gr.Textbox(label="Feature Summary",
                                                   lines=8, interactive=False)
                        download_btn = gr.File(label="Download Enriched CSV")

                with gr.Row():
                    preview_table = gr.Dataframe(
                        label="Output Preview (first 20 rows)", interactive=False
                    )
                with gr.Row():
                    log_box = gr.Textbox(label="Agent Logs", lines=20, interactive=False)

            # ── History ────────────────────────────────────────────────────────
            with gr.Tab("History"):
                refresh_btn  = gr.Button("Refresh", variant="secondary")
                history_html = gr.HTML(value=load_history_html())

                gr.Markdown("### Resume a failed run")
                with gr.Row():
                    failed_dropdown = gr.Dropdown(
                        choices=load_failed_runs(),
                        label="Failed runs",
                        info="Select a run to resume from its last checkpoint",
                        scale=4,
                    )
                    resume_btn = gr.Button("Resume", variant="primary", scale=1)

                resume_logs     = gr.Textbox(label="Resume Logs",     lines=10, interactive=False)
                resume_summary  = gr.Textbox(label="Feature Summary", lines=6,  interactive=False)
                resume_preview  = gr.Dataframe(label="Output Preview", interactive=False)
                resume_download = gr.File(label="Download Enriched CSV")

                async def _resume_from_dropdown(selection: str):
                    import gradio as _gr
                    if not selection:
                        raise _gr.Error("Select a failed run first.")
                    tid = selection.split("|")[-1].strip()
                    return await resume_run(tid)

                refresh_btn.click(
                    fn=lambda: (load_history_html(),
                                gr.Dropdown(choices=load_failed_runs())),
                    inputs=[],
                    outputs=[history_html, failed_dropdown],
                )
                resume_btn.click(
                    fn=_resume_from_dropdown,
                    inputs=[failed_dropdown],
                    outputs=[resume_logs, resume_summary,
                             resume_preview, resume_download, history_html],
                )

        run_btn.click(
            fn=run_agent,
            inputs=[csv_input, hint_input, max_features_slider, exclude_input],
            outputs=[log_box, summary_box, preview_table, download_btn, history_html],
        )

    demo.launch(share=False)
