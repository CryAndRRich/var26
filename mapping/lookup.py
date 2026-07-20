"""Mapping baseline (CPU): tra cứu mã đã học từ train + GATE + fallback exact ICD.

Insight data: >80% CHẨN_ĐOÁN & ~70% THUỐC để candidates=[]; gần như luôn 1 mã.
=> GATE (có gán mã không) quan trọng nhất. Baseline này học GATE từ train:
   với mỗi mention, xác suất được gán mã và mã phổ biến nhất.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..io.schema import Concept, TYPES_WITH_CANDIDATES
from .icd_index import ICDIndex


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class LookupMapper:
    def __init__(
        self,
        mention2code: dict[tuple[str, str], str],
        icd: ICDIndex | None = None,
        use_exact_icd: bool = True,
    ):
        # khóa: (type, mention_chuẩn_hóa) -> mã phổ biến nhất (chỉ khi train hay gán mã)
        self.mention2code = mention2code
        self.icd = icd
        self.use_exact_icd = use_exact_icd

    @classmethod
    def fit(
        cls,
        labeled: list[list[Concept]],
        icd: ICDIndex | None = None,
        min_code_ratio: float = 0.5,
    ) -> "LookupMapper":
        """Học GATE + mã. Chỉ nhớ mã cho mention mà >= min_code_ratio lần có mã trong train."""
        code_votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        seen: dict[tuple[str, str], int] = defaultdict(int)
        with_code: dict[tuple[str, str], int] = defaultdict(int)
        for concepts in labeled:
            for c in concepts:
                if c.type not in TYPES_WITH_CANDIDATES:
                    continue
                k = (c.type, _key(c.text))
                seen[k] += 1
                if c.candidates:
                    with_code[k] += 1
                    # nhớ mã đầu (đa số 1 mã)
                    code_votes[k][c.candidates[0]] += 1
        mention2code: dict[tuple[str, str], str] = {}
        for k, n in seen.items():
            if with_code[k] / n >= min_code_ratio and code_votes[k]:
                mention2code[k] = code_votes[k].most_common(1)[0][0]
        return cls(mention2code, icd=icd)

    def map_concept(self, text: str, concept: Concept) -> list[str]:
        if concept.type not in TYPES_WITH_CANDIDATES:
            return []
        k = (concept.type, _key(concept.text))
        # 1) mã đã học (đã qua GATE khi fit)
        if k in self.mention2code:
            return [self.mention2code[k]]
        # 2) fallback exact ICD cho CHẨN_ĐOÁN (precision cao, recall thấp)
        if concept.type == "CHẨN_ĐOÁN" and self.use_exact_icd and self.icd:
            hit = self.icd.exact(concept.text)
            if len(hit) == 1:  # chỉ nhận khi duy nhất -> tránh nhiễu
                return hit
        # 3) GATE: mặc định để rỗng (đa số GT rỗng)
        return []
