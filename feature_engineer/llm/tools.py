"""
LangChain Tool wrappers for external APIs.
Each tool is independently testable by mocking the underlying HTTP call.
"""
import requests
from langchain.agents import Tool

from feature_engineer.config import SERPER_KEY
from feature_engineer.llm.setup import llm


def _serper_search(query: str) -> str:
    """Execute a Serper web search. Returns concatenated snippets or fallback message."""
    if not SERPER_KEY:
        return "No SERPER_API_KEY set — no web results available."
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        resp.raise_for_status()
        items    = resp.json().get("organic", [])[:5]
        snippets = [
            f"{r.get('title', '')}: {r.get('snippet', '')}"
            for r in items if r.get("snippet")
        ]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


serper_tool = Tool(
    name="web_search",
    func=_serper_search,
    description=(
        "Search the web for feature engineering best practices for a given ML use case. "
        "Input: a short search query string."
    ),
)
