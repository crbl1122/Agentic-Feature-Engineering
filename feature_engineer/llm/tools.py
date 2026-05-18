"""
LangChain Tool wrappers for external APIs.
Each tool is independently testable by mocking the underlying HTTP call.
"""
import requests
from langchain.agents import Tool

from feature_engineer.config import SERPER_KEY
from feature_engineer.llm.setup import llm


def _serper_search(query: str) -> str:
    """Execute a Serper web search. Returns titles, links and snippets."""
    print(f"[serper] Query: {query}")
    if not SERPER_KEY:
        return "No SERPER_API_KEY set — no web results available."
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 8},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("organic", [])[:8]
        lines = []
        for r in items:
            title   = r.get("title", "")
            link    = r.get("link", "")
            snippet = r.get("snippet", "")
            if title or snippet:
                lines.append(f"{title}: {snippet}")
                if link:
                    lines.append(f"URL: {link}")
        return "\n".join(lines) if lines else "No results found."
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
