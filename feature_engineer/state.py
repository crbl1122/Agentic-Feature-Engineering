"""
State definitions — AgentState and all Pydantic models.
No imports from other feature_engineer modules (no circular deps).
"""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── Pydantic models ─────────────────────────────────────────────────────────────

class FeaturePlan(BaseModel):
    """LLM-produced specification for a single feature column."""
    feature_name: str = Field(description="Snake-case name for the new column")
    description:  str = Field(description="One-line plain-English description")
    pandas_code:  str = Field(
        description=(
            "A single Python expression (no assignments, no imports) that "
            "takes `df` and returns a pandas Series. "
            "Example: df['col_a'] / (df['col_b'] + 1)"
        )
    )


class FeaturePlanList(BaseModel):
    """LLM-produced list of feature plans."""
    features: list[FeaturePlan] = Field(
        description="List of feature plans, one per new column."
    )


class FilteredPlanList(BaseModel):
    """LLM-filtered list of feature plans that respect the user hint."""
    kept:    list[FeaturePlan] = Field(description="Features that respect the hint")
    dropped: list[str]         = Field(description="Dropped feature_names and reasons")


class FeatureVerdict(BaseModel):
    """Chain-of-thought verdict for a single feature."""
    feature_name:  str = Field(description="Name of the feature")
    grouping_col:  str = Field(description="Column used for grouping/filtering, or 'none'")
    date_usage:    str = Field(description="How date/time columns are used, or 'none'")
    leakage_check: str = Field(description="Leakage analysis and result")
    hint_check:    str = Field(description="Hint compliance analysis, or 'no hint'")
    verdict:       str = Field(description="'keep' or 'drop'")
    drop_reason:   str = Field(description="Reason for dropping, empty string if kept")


class VerdictList(BaseModel):
    """Chain-of-thought verdicts for all planned features."""
    verdicts: list[FeatureVerdict]


# ── LangGraph state ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    input_path:          str
    output_path:         str
    objective:           str                                          # use case + constraints
    max_features:        int
    df:                  str                                          # path to parquet
    original_columns:    list[str]
    plan:                FeaturePlan | None
    feature_queue:       list[dict]
    feature_candidates:  list[str]
    feasible_features:   list[str]
    research_messages:   Annotated[list, add_messages]                # ReAct loop messages
    completed_features:  Annotated[list[str],  lambda a, b: a + b]
    completed_plans:     Annotated[list[dict], lambda a, b: a + b]
    completed_formulas:  Annotated[list[str],  lambda a, b: a + b]   # validated formulas
    failed_formulas:     Annotated[list[str],  lambda a, b: a + b]   # failed formulas
    errors:              Annotated[list[str],  lambda a, b: b]        # replace reducer
    attempts:            int


def empty_state(
    input_path: str,
    output_path: str,
    objective: str,
    max_features: int,
) -> AgentState:
    """Return a fully initialised AgentState ready to invoke."""
    return AgentState(
        input_path=input_path,
        output_path=output_path,
        objective=objective,
        max_features=max_features,
        df="",
        original_columns=[],
        plan=None,
        feature_queue=[],
        feature_candidates=[],
        feasible_features=[],
        research_messages=[],
        completed_features=[],
        completed_plans=[],
        completed_formulas=[],
        failed_formulas=[],
        errors=[],
        attempts=0,
    )
