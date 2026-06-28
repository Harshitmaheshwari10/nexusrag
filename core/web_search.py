"""
NexusRAG Web Searcher
Uses DuckDuckGo Search (duckduckgo-search package) — 100% free, no API key.
"""

from typing import List, Dict, Any
import re


class WebSearcher:
    """Searches DuckDuckGo and returns structured results."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Returns list of {title, snippet, url}."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            cleaned = []
            for r in results:
                cleaned.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })
            return cleaned
        except ImportError:
            return [{"title": "Error", "snippet": "Install duckduckgo-search: pip install duckduckgo-search", "url": ""}]
        except Exception as e:
            # Graceful fallback
            return [{"title": "Search unavailable", "snippet": str(e), "url": ""}]

    def format_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format web results into a context string for the LLM."""
        if not results:
            return "No web results found."

        lines = ["[WEB SEARCH RESULTS]\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"Result {i}: {r['title']}")
            if r["url"]:
                lines.append(f"URL: {r['url']}")
            lines.append(f"Content: {r['snippet']}\n")
        return "\n".join(lines)
