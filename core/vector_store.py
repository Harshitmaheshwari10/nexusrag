"""
NexusRAG Vector Store
- Dense retrieval: FAISS + sentence-transformers/all-MiniLM-L6-v2 (free, local)
- Sparse retrieval: BM25 (rank_bm25)
- Hybrid fusion: Reciprocal Rank Fusion (RRF)
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    text: str
    score: float
    doc_name: str
    chunk_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreManager:
    def __init__(self):
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._embedding_model = None
        self._faiss_index = None
        self._bm25 = None

    # ── Lazy-load embedding model ─────────────────────────────────────────────
    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError("Run: pip install sentence-transformers")
        return self._embedding_model

    # ── Add documents ─────────────────────────────────────────────────────────
    def add_documents(
        self, chunks: List[str], metadata: Dict[str, Any], doc_name: str
    ) -> List[int]:
        model = self._get_embedding_model()

        start_idx = len(self.chunks)
        new_metas = [
            {"doc_name": doc_name, "chunk_idx": i + start_idx, **metadata}
            for i in range(len(chunks))
        ]

        self.chunks.extend(chunks)
        self.chunk_meta.extend(new_metas)

        # Embed new chunks
        new_embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Rebuild FAISS index
        self._rebuild_faiss()

        # Rebuild BM25
        self._rebuild_bm25()

        return list(range(start_idx, len(self.chunks)))

    def _rebuild_faiss(self):
        try:
            import faiss
            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim (with normalized vecs)
            index.add(self.embeddings.astype("float32"))
            self._faiss_index = index
        except ImportError:
            # Fallback: pure numpy cosine
            self._faiss_index = None

    def _rebuild_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [c.lower().split() for c in self.chunks]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    # ── Search ─────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.chunks:
            return []

        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)

        if not dense_results and not sparse_results:
            return []

        if not sparse_results:
            return dense_results[:top_k]

        if not dense_results:
            return sparse_results[:top_k]

        # Reciprocal Rank Fusion
        fused = self._rrf_fusion(dense_results, sparse_results, top_k)
        return fused

    def _dense_search(self, query: str, top_k: int) -> List[SearchResult]:
        if not self.chunks or self.embeddings is None:
            return []

        model = self._get_embedding_model()
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_emb, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    meta = self.chunk_meta[idx]
                    results.append(SearchResult(
                        text=self.chunks[idx],
                        score=float(score),
                        doc_name=meta.get("doc_name", "unknown"),
                        chunk_idx=idx,
                        metadata=meta,
                    ))
            return results
        else:
            # Numpy fallback
            sims = self.embeddings @ q_emb.T
            sims = sims.flatten()
            top_idxs = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_idxs:
                meta = self.chunk_meta[idx]
                results.append(SearchResult(
                    text=self.chunks[idx],
                    score=float(sims[idx]),
                    doc_name=meta.get("doc_name", "unknown"),
                    chunk_idx=int(idx),
                    metadata=meta,
                ))
            return results

    def _sparse_search(self, query: str, top_k: int) -> List[SearchResult]:
        if self._bm25 is None or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idxs:
            if scores[idx] > 0:
                meta = self.chunk_meta[idx]
                results.append(SearchResult(
                    text=self.chunks[idx],
                    score=float(scores[idx]),
                    doc_name=meta.get("doc_name", "unknown"),
                    chunk_idx=int(idx),
                    metadata=meta,
                ))
        return results

    def _rrf_fusion(
        self,
        dense: List[SearchResult],
        sparse: List[SearchResult],
        top_k: int,
        k: int = 60,
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion — combines dense and sparse rankings."""
        scores: Dict[int, float] = {}
        idx_to_result: Dict[int, SearchResult] = {}

        for rank, r in enumerate(dense):
            scores[r.chunk_idx] = scores.get(r.chunk_idx, 0) + 1.0 / (k + rank + 1)
            idx_to_result[r.chunk_idx] = r

        for rank, r in enumerate(sparse):
            scores[r.chunk_idx] = scores.get(r.chunk_idx, 0) + 1.0 / (k + rank + 1)
            if r.chunk_idx not in idx_to_result:
                idx_to_result[r.chunk_idx] = r

        sorted_idxs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        results = []
        for idx in sorted_idxs:
            r = idx_to_result[idx]
            r.score = scores[idx]  # Replace with RRF score
            results.append(r)
        return results

    def has_documents(self) -> bool:
        return len(self.chunks) > 0
