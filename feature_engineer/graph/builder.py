"""
Graph assembly — wires all nodes and edges into a StateGraph.
Returns uncompiled StateGraph; caller adds checkpointer.
"""
from langgraph.graph import END, START, StateGraph

from feature_engineer.nodes.execution import (
    create_feature, record_feature, validate, validate_code,
)
from feature_engineer.nodes.ingestion import load_csv
from feature_engineer.nodes.planning import feature_planner, validate_plan
from feature_engineer.nodes.research import (
    extract_candidates, map_to_columns, research_features,
    research_tool_node, research_tools_condition,
    evaluate_research, after_evaluate_research,
)
from feature_engineer.nodes.revision import revise_plan
from feature_engineer.nodes.routing import (
    after_next, after_record, next_feature, save_csv,
    should_execute, should_retry,
)
from feature_engineer.state import AgentState


def build_graph() -> StateGraph:
    """Build and return the uncompiled feature engineering StateGraph."""
    g = StateGraph(AgentState)

    # ── nodes ──────────────────────────────────────────────────────────────────
    g.add_node("load_csv",           load_csv)
    g.add_node("research_features",  research_features)
    g.add_node("research_tools",     research_tool_node)
    g.add_node("extract_candidates", extract_candidates)
    g.add_node("evaluate_research",  evaluate_research)
    g.add_node("map_to_columns",     map_to_columns)
    g.add_node("feature_planner",    feature_planner)
    g.add_node("validate_plan",      validate_plan)
    g.add_node("validate_code",      validate_code)
    g.add_node("create_feature",     create_feature)
    g.add_node("validate",           validate)
    g.add_node("record_feature",     record_feature)
    g.add_node("revise_plan",        revise_plan)
    g.add_node("next_feature",       next_feature)
    g.add_node("save_csv",           save_csv)

    # ── unconditional edges ────────────────────────────────────────────────────
    g.add_edge(START,                "load_csv")
    g.add_edge("load_csv",           "research_features")
    g.add_edge("research_tools",     "research_features")
    g.add_edge("extract_candidates", "evaluate_research")   # evaluate before mapping
    g.add_edge("map_to_columns",     "feature_planner")
    g.add_edge("feature_planner",    "validate_plan")
    g.add_edge("validate_plan",      "validate_code")
    g.add_edge("create_feature",     "validate")
    g.add_edge("revise_plan",        "validate_code")
    g.add_edge("save_csv",           END)

    # ── conditional edges ──────────────────────────────────────────────────────
    g.add_conditional_edges(
        "research_features",
        research_tools_condition,
        {"tools": "research_tools", "done": "extract_candidates"},
    )
    g.add_conditional_edges(
        "evaluate_research",
        after_evaluate_research,
        {"retry": "research_features", "continue": "map_to_columns"},
    )
    g.add_conditional_edges(
        "validate_code",
        should_execute,
        {"run": "create_feature", "revise": "revise_plan",
         "next": "next_feature",  "save": "save_csv"},
    )
    g.add_conditional_edges(
        "validate",
        should_retry,
        {"revise": "revise_plan", "record": "record_feature",
         "next": "next_feature",  "save": "save_csv"},
    )
    g.add_conditional_edges(
        "record_feature",
        after_record,
        {"next": "next_feature", "save": "save_csv"},
    )
    g.add_conditional_edges(
        "next_feature",
        after_next,
        {"run": "validate_code", "save": "save_csv", "revise": "revise_plan"},
    )

    # diagram (no checkpointer needed)
    preview = g.compile()
    try:
        with open("graph.png", "wb") as f:
            f.write(preview.get_graph().draw_mermaid_png())
        print("[build_graph] Graph diagram saved to graph.png")
    except Exception:
        pass

    return g
