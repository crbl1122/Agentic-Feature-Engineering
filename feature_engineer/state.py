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

class ResearchEvaluatorOutput(BaseModel):
    """Structured output for the research evaluator."""
    feedback:    str  = Field(description="What is generic/wrong and what to search for instead")
    is_specific: bool = Field(description="True if candidates are domain-specific feature names, not methodology categories")
    retry_query: str  = Field(description="A better, more specific search query for the next attempt")


class ResearchFeature(BaseModel):
    """A single feature candidate produced by the research loop."""
    name:         str = Field(description="Short snake_case feature name")
    description:  str = Field(description="One-line description of what it measures")
    formula_hint: str = Field(
        description=(
            "A pandas expression hint using the actual column names available. "
            "Example: df['col_a'] / df['col_b']"
        )
    )


class ResearchFeatureList(BaseModel):
    """Curated list of feature candidates from research."""
    features: list[ResearchFeature] = Field(
        description="List of specific, computable feature candidates."
    )


class RejectionReason(BaseModel):
    """Reason for rejecting a single candidate."""
    candidate: str = Field(description="The rejected candidate name")
    reason:    str = Field(description="Why it was rejected")


class EvaluationResult(BaseModel):
    """Structured output for the candidate evaluator."""
    specific:          list[str]            = Field(description="Candidates that are specific, domain-relevant and computable")
    generic:           list[str]            = Field(description="Candidates that are vague, generic, off-domain or identical to existing columns")
    rejection_reasons: list[RejectionReason] = Field(description="Rejection reason per generic candidate", default_factory=list)


class FeasibleFeature(BaseModel):
    """A single feasible feature after column mapping."""
    name:        str       = Field(description="Short snake_case feature name")
    description: str       = Field(description="What it measures")
    needs:       list[str] = Field(description="Column names required to compute this feature")
    formula_hint: str      = Field(description="Pandas expression hint using available columns", default="")


class FeasibleFeatureList(BaseModel):
    """Features that can be computed from available CSV columns."""
    features: list[FeasibleFeature] = Field(
        description="Only features that can be meaningfully computed from available columns."
    )


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
    objective:           str
    max_features:        int
    df:                  str
    original_columns:    list[str]
    plan:                FeaturePlan | None
    feature_queue:       list[dict]
    feature_candidates:  list[str]
    feasible_features:   list[str]
    good_candidates:     list[str]
    research_formula_hints: dict                                      # {feature_label: formula_hint}
    research_messages:   Annotated[list, add_messages]
    research_attempts:   int
    research_feedback:   str
    research_is_specific: bool                                        # evaluator verdict
    completed_features:  Annotated[list[str],  lambda a, b: a + b]
    completed_plans:     Annotated[list[dict], lambda a, b: a + b]
    completed_formulas:  Annotated[list[str],  lambda a, b: a + b]
    failed_formulas:     Annotated[list[str],  lambda a, b: a + b]
    errors:              Annotated[list[str],  lambda a, b: b]
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
        good_candidates=[],
        research_formula_hints={},
        research_messages=[],
        research_attempts=0,
        research_feedback="",
        research_is_specific=False,
        completed_features=[],
        completed_plans=[],
        completed_formulas=[],
        failed_formulas=[],
        errors=[],
        attempts=0,
    )
