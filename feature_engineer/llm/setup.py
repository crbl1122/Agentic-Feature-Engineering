"""
LLM instances — single source of truth for all model configurations.
Structured output variants live here so nodes stay free of LangChain setup.
"""
from langchain_openai import ChatOpenAI

from feature_engineer.config import LLM_MODEL
from feature_engineer.state import FeaturePlan, FeaturePlanList, VerdictList

# base model — used everywhere
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# structured output variants
structured_llm = llm.with_structured_output(FeaturePlan)
planner_llm    = llm.with_structured_output(FeaturePlanList)
verdict_llm    = llm.with_structured_output(VerdictList)
