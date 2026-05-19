"""
Custom arXiv MCP server — search, download, and read academic papers.
Uses arXiv public API (no credentials required).
"""
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv-custom")

PAPERS_DIR = Path(__file__).parent.parent / "storage" / "arxiv_papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv papers by query and return title, abstract and pdf_url."""
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": query, "max_results": max_results},
        timeout=30,
    )
    root = ET.fromstring(response.text)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text.strip().split("/")[-1]
        papers.append({
            "id":       arxiv_id,
            "title":    entry.find("{http://www.w3.org/2005/Atom}title").text.strip(),
            "abstract": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
            "pdf_url":  f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return papers


@mcp.tool()
def download_paper(paper_id: str, pdf_url: str) -> str:
    """Download a paper PDF from arXiv and save it locally. Returns the local file path."""
    path = PAPERS_DIR / f"{paper_id.replace('/', '_')}.pdf"
    if path.exists():
        return str(path)
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return str(path)


@mcp.tool()
def read_paper(paper_id: str) -> str:
    """Read the text content of a previously downloaded paper."""
    path = PAPERS_DIR / f"{paper_id.replace('/', '_')}.pdf"
    if not path.exists():
        return f"Paper {paper_id} not found. Download it first."
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


if __name__ == "__main__":
    mcp.run(transport="stdio")
