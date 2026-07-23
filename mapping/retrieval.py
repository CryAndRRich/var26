"""Candidate retrieval cho code-mapping (EXP4) — tầng 2 (khi exact fail).

- BM25Retriever: lexical, thuần Python (rank_bm25), CHẠY ĐƯỢC TRÊN CPU local.
- DenseRetriever: SapBERT/bge-m3, nhận embedding đã tính sẵn (encode ở notebook GPU).
- HybridRetriever: hợp nhất bằng Reciprocal Rank Fusion (RRF).

KB entry là object có .code / .description / .norm (ICDEntry, RxNormEntry đều hợp).
Trả top-k dạng [(code, score, description)] đã KHỬ TRÙNG code (giữ điểm cao nhất).
"""
from __future__ import annotations

from .icd_index import normalize


def _dedup_by_code(ranked: list[tuple[str, float, str]]) -> list[tuple[str, float, str]]:
    best: dict[str, tuple[str, float, str]] = {}
    for code, score, desc in ranked:
        if code not in best or score > best[code][1]:
            best[code] = (code, score, desc)
    return sorted(best.values(), key=lambda x: -x[1])


def _tokenize(s: str) -> list[str]:
    return normalize(s).split()


class BM25Retriever:
    """BM25 trên description của KB. Lexical, không cần GPU."""

    def __init__(self, entries):
        from rank_bm25 import BM25Okapi

        self.entries = list(entries)
        self._corpus_tokens = [_tokenize(e.description) for e in self.entries]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, mention: str, k: int = 20) -> list[tuple[str, float, str]]:
        scores = self.bm25.get_scores(_tokenize(mention))
        idx = sorted(range(len(scores)), key=lambda i: -scores[i])[: k * 3]
        ranked = [(self.entries[i].code, float(scores[i]), self.entries[i].description) for i in idx]
        return _dedup_by_code(ranked)[:k]


class DenseRetriever:
    """Dense retrieval bằng embedding đã tính sẵn (cosine).

    `embeddings`: ma trận (N, d) đã chuẩn hóa L2, khớp thứ tự `entries`.
    `encode_fn(list[str]) -> (M, d) L2-normalized` để encode mention lúc query
    (đặt ở notebook, bọc SapBERT/bge-m3). Giữ module này không phụ thuộc torch.
    """

    def __init__(self, entries, embeddings, encode_fn):
        self.entries = list(entries)
        self.embeddings = embeddings  # numpy (N,d)
        self.encode_fn = encode_fn

    def search(self, mention: str, k: int = 20) -> list[tuple[str, float, str]]:
        import numpy as np

        q = np.asarray(self.encode_fn([mention]))[0]
        sims = self.embeddings @ q
        idx = np.argsort(-sims)[: k * 3]
        ranked = [(self.entries[i].code, float(sims[i]), self.entries[i].description) for i in idx]
        return _dedup_by_code(ranked)[:k]


class HybridRetriever:
    """Hợp nhất nhiều retriever bằng RRF (không phụ thuộc thang điểm)."""

    def __init__(self, retrievers: list, rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, mention: str, k: int = 20) -> list[tuple[str, float, str]]:
        fused: dict[str, float] = {}
        desc: dict[str, str] = {}
        for r in self.retrievers:
            for rank, (code, _score, d) in enumerate(r.search(mention, k=k)):
                fused[code] = fused.get(code, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                desc.setdefault(code, d)
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(code, score, desc[code]) for code, score in ranked]
