"""
LLM instances — single source of truth for all model configurations.
Structured output variants live here so nodes stay free of LangChain setup.
"""
from langchain_openai import ChatOpenAI

from feature_engineer.config import LLM_MODEL
from feature_engineer.state import (
    FeaturePlan, FeaturePlanList, VerdictList,
    ResearchEvaluatorOutput, ResearchFeatureList,
    EvaluationResult, FeasibleFeatureList,
)

# base model — deterministic for planning, validation, revision
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# structured output variants
structured_llm          = llm.with_structured_output(FeaturePlan)
planner_llm             = llm.with_structured_output(FeaturePlanList)
verdict_llm             = llm.with_structured_output(VerdictList)
research_evaluator_llm  = llm.with_structured_output(ResearchEvaluatorOutput)
evaluation_llm          = llm.with_structured_output(EvaluationResult)
mapping_llm             = llm.with_structured_output(FeasibleFeatureList)

# research LLM — slightly creative for diverse feature discovery
research_base_llm         = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
research_structured_llm   = research_base_llm.with_structured_output(ResearchFeatureList)
