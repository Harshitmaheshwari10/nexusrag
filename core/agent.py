"""
NexusRAG Agent
- Groq API (free tier: llama-3.3-70b-versatile, 14,400 req/day)
- Smart routing: docs / web / hybrid
- Answer synthesis with source attribution
- Conversation-aware (uses chat history)
"""

import re
import json
from typing import List, Dict, Any, Optional
from core.vector_store import VectorStoreManager, SearchResult
from core.web_search import WebSearcher


SYSTEM_PROMPT = """You are NexusRAG, an intelligent research assistant that answers questions using retrieved context from documents and/or web search results.

Your behavior:
1. Always ground your answer in the provided context
2. If context is from documents, cite the document name
3. If context is from the web, mention it's from web search
4. Be concise but comprehensive
5. Use markdown formatting for clarity (bold key terms, bullet points for lists)
6. If the context doesn't contain enough information, say so clearly
7. Never make up information not present in the context

Format your answer in clean markdown. Be helpful and precise."""

ROUTER_PROMPT = """You are a query router. Decide the best source to answer this query.

Query: {query}
Documents available: {has_docs}
Document topics: {doc_topics}

Respond with EXACTLY one word:
- "docs" if the query is about content in the uploaded documents
- "web" if the query needs recent/real-time info or is about general knowledge not in docs
- "hybrid" if both sources would help

Your response (one word only):"""


class NexusAgent:
    def __init__(self, groq_key: str, vector_store: VectorStoreManager):
        self.groq_key = groq_key
        self.vector_store = vector_store
        self.web_searcher = WebSearcher(max_results=5)
        self._groq_client = None

    def _get_client(self):
        if self._groq_client is None:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_key)
            except ImportError:
                raise ImportError("Run: pip install groq")
        return self._groq_client

    def _chat(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _route_query(self, query: str, mode: str) -> str:
        """Decide whether to search docs, web, or both."""
        if mode in ("docs", "web"):
            return mode

        # Auto-route in hybrid mode
        has_docs = self.vector_store.has_documents()
        if not has_docs:
            return "web"

        # Use keyword heuristics first (fast)
        web_keywords = r'\b(today|yesterday|latest|current|recent|now|2024|2025|2026|news|price|weather|live|update)\b'
        if re.search(web_keywords, query.lower()):
            return "hybrid"

        # For short queries about general knowledge, prefer web
        if len(query.split()) < 4 and not has_docs:
            return "web"

        # Default to docs if we have them
        return "docs" if has_docs else "web"

    def run(
        self,
        query: str,
        mode: str = "hybrid",
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Main entry point. Returns {answer, sources, web_results, source_type}."""

        routing = self._route_query(query, mode)

        doc_context = ""
        web_context = ""
        doc_sources = []
        web_results = []

        # ── Retrieve from docs ────────────────────────────────────────────────
        if routing in ("docs", "hybrid"):
            if self.vector_store.has_documents():
                results: List[SearchResult] = self.vector_store.search(query, top_k=5)
                doc_sources = [
                    {
                        "text": r.text[:300] + ("..." if len(r.text) > 300 else ""),
                        "doc_name": r.doc_name,
                        "score": round(r.score, 4),
                        "chunk_idx": r.chunk_idx,
                    }
                    for r in results
                ]
                if results:
                    doc_context = "[DOCUMENT CONTEXT]\n\n" + "\n\n---\n\n".join(
                        f"Source: {r.doc_name}\n{r.text}" for r in results
                    )

        # ── Retrieve from web ─────────────────────────────────────────────────
        if routing in ("web", "hybrid"):
            web_results = self.web_searcher.search(query)
            if web_results:
                web_context = self.web_searcher.format_for_context(web_results)

        # ── Build context ─────────────────────────────────────────────────────
        if doc_context and web_context:
            full_context = doc_context + "\n\n" + web_context
        elif doc_context:
            full_context = doc_context
        elif web_context:
            full_context = web_context
        else:
            full_context = "No relevant context found in documents or web search."

        # ── Build messages ────────────────────────────────────────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add last 4 exchanges of history
        if chat_history:
            for h in chat_history[-8:]:
                messages.append({"role": h["role"], "content": h["content"]})

        user_content = f"""Context:
{full_context}

Question: {query}

Please answer based on the context above. If using document sources, mention which document. If using web results, say "according to web search"."""

        messages.append({"role": "user", "content": user_content})

        # ── Generate answer ───────────────────────────────────────────────────
        answer = self._chat(messages, temperature=0.1, max_tokens=1024)

        return {
            "answer": answer,
            "sources": doc_sources,
            "web_results": web_results,
            "source_type": routing,
        }
