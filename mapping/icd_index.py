"""Nạp & tra cứu ICD-10 từ icd10_map.csv (CPU, không cần GPU).

Dùng cho candidate generation tầng 1 (exact / near-exact match).
Tầng dense (SapBERT) và LLM rerank cài ở module riêng (chạy trên notebooks/).
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def normalize(s: str) -> str:
    """Chuẩn hóa để so khớp: hạ chữ, bỏ dấu câu thừa, gộp khoảng trắng.
    (Chỉ dùng cho MATCHING — không dùng để tính offset.)"""
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


@dataclass
class ICDEntry:
    code: str
    description: str
    norm: str


class ICDIndex:
    def __init__(self, entries: list[ICDEntry]):
        self.entries = entries
        self._by_norm: dict[str, list[str]] = {}
        for e in entries:
            self._by_norm.setdefault(e.norm, []).append(e.code)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ICDIndex":
        entries: list[ICDEntry] = []
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                code = (row.get("code") or "").strip()
                desc = (row.get("description") or "").strip()
                if code:
                    entries.append(ICDEntry(code, desc, normalize(desc)))
        return cls(entries)

    def exact(self, mention: str) -> list[str]:
        """Khớp chính xác (sau normalize) description -> list code."""
        return self._by_norm.get(normalize(mention), [])

    def __len__(self) -> int:
        return len(self.entries)

# TODO(BƯỚC 4+): thêm BM25 (rank_bm25) trên self.entries + dense SapBERT retrieval,
# rồi LLM rerank. Interface đề xuất: candidate_search(mention, k) -> list[(code, score)].
