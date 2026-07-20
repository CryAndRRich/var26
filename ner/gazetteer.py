"""Gazetteer NER (CPU) — baseline: học phrase->type từ train_labels, khớp chuỗi.

Không cần GPU. Dùng làm baseline end-to-end trước khi có encoder.
- Học: gom mọi (text_chuẩn_hóa -> type phổ biến nhất) từ dữ liệu có nhãn.
- Gán nhãn: tìm mọi lần xuất hiện của phrase đã biết trong text, chọn tập span
  KHÔNG chồng lấn theo chiến lược longest-match (span dài ưu tiên).
- position khớp tuyệt đối input[start:end]; `text` = lát cắt RAW từ input.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..io.schema import Concept


def _norm_key(s: str) -> str:
    """Khóa so khớp: hạ chữ + gộp khoảng trắng. (KHÔNG bỏ dấu để tránh nhập nhằng.)"""
    return re.sub(r"\s+", " ", s.strip().lower())


class GazetteerTagger:
    def __init__(self, phrase2type: dict[str, str]):
        # loại bỏ phrase quá ngắn/nhiễu (1 ký tự)
        self.phrase2type = {p: t for p, t in phrase2type.items() if len(p) >= 2}
        # gom theo độ dài giảm dần để ưu tiên longest-match
        self._phrases_by_len = sorted(
            self.phrase2type.keys(), key=len, reverse=True
        )

    @classmethod
    def fit(cls, labeled: list[list[Concept]]) -> "GazetteerTagger":
        """labeled: list các sample, mỗi sample là list Concept (ground truth)."""
        votes: dict[str, Counter] = defaultdict(Counter)
        for concepts in labeled:
            for c in concepts:
                key = _norm_key(c.text)
                if key:
                    votes[key][c.type] += 1
        phrase2type = {k: cnt.most_common(1)[0][0] for k, cnt in votes.items()}
        return cls(phrase2type)

    def tag(self, text: str) -> list[Concept]:
        low = text.lower()
        occupied = [False] * len(text)
        found: list[Concept] = []
        for phrase in self._phrases_by_len:
            ptype = self.phrase2type[phrase]
            start = 0
            while True:
                idx = low.find(phrase, start)
                if idx < 0:
                    break
                end = idx + len(phrase)
                # chỉ nhận nếu vùng chưa bị chiếm và biên là ranh giới từ/không chữ
                if not any(occupied[idx:end]) and self._boundary_ok(low, idx, end):
                    found.append(
                        Concept(text=text[idx:end], type=ptype, position=(idx, end))
                    )
                    for i in range(idx, end):
                        occupied[i] = True
                start = idx + 1
        found.sort(key=lambda c: c.position[0])
        return found

    @staticmethod
    def _boundary_ok(low: str, idx: int, end: int) -> bool:
        """Tránh khớp giữa từ (vd 'ho' trong 'khó thở'). Kiểm tra ký tự biên."""
        before = low[idx - 1] if idx > 0 else " "
        after = low[end] if end < len(low) else " "
        return not (before.isalnum() or after.isalnum())

    def __len__(self) -> int:
        return len(self.phrase2type)
