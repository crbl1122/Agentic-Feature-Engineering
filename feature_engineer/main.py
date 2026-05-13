"""
CLI entry point — thin wrapper that delegates to ui.app or runs graph directly.
"""
import argparse
import asyncio
import os
import uuid

from feature_engineer.config import DEFAULT_MAX_FEATURES, DEFAULT_RECURSION_LIMIT, DB_PATH, OUTPUT_DIR
from feature_engineer.graph.builder import build_graph
from feature_engineer.state import empty_state
from feature_engineer.storage.database import init_db, save_run
from feature_engineer.storage.parquet import init_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph feature engineering agent")
    parser.add_argument("--input",        default="",  help="Path to input CSV")
    parser.add_argument("--output",       default="",  help="Path to write enriched CSV")
    parser.add_argument("--objective",    default="",  help="ML objective and feature constraints")
    parser.add_argument("--max-features", default=DEFAULT_MAX_FEATURES, type=int)
    parser.add_argument("--thread-id",    default="",  help="Resume a specific checkpoint")
    parser.add_argument("--ui",           action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()

    if args.ui:
        from feature_engineer.ui.app import launch_ui
        launch_ui()
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required when not using --ui")

    asyncio.run(_run_cli(args))


async def _run_cli(args) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    init_db()
    init_dirs()

    thread_id = args.thread_id or str(uuid.uuid4())
    config    = {
        "configurable":    {"thread_id": thread_id},
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }
    print(f"[main] thread_id: {thread_id}")

    initial_state = empty_state(
        input_path=args.input,
        output_path=args.output,
        objective=args.objective,
        max_features=args.max_features,
    )

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph       = build_graph().compile(checkpointer=checkpointer)
        final_state = await graph.ainvoke(initial_state, config=config)

    save_run(thread_id, final_state, "success")

    completed = final_state.get("completed_features", [])
    print("\n── Summary ────────────────────────────────")
    print(f"  Thread    : {thread_id}")
    print(f"  Features  : {completed}")
    print(f"  Output    : {args.output}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
