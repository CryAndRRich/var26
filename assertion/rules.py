"""Assertion tiếng Việt bằng rule (NegEx/ConText Việt hóa) — precision-first.

Chỉ áp cho CHẨN_ĐOÁN / THUỐC / TRIỆU_CHỨNG. Phân bố lệch mạnh
(isHistorical 1085 · isNegated 567 · isFamily 4); mỗi concept 0–1 assertion.
Đoán thừa assertion -> 0 điểm sample đó => ưu tiên PRECISION.

Ý tưởng chính (khác bản đầu): xét cue trong **MỆNH ĐỀ** chứa concept
(bounded bởi xuống dòng, gạch đầu dòng, dấu . ; :) thay vì cửa sổ ký tự cố định.
Cue phải đứng TRƯỚC mention trong cùng mệnh đề.
"""
from __future__ import annotations

import re

from ..io.schema import Concept, TYPES_WITH_ASSERTIONS

# Cue phủ định — CHỈ cue mạnh, ít nhập nhằng (bỏ "không"/"chưa" trần vì over-fire).
NEGATION_CUES = [
    "không có", "không thấy", "không ghi nhận", "chưa ghi nhận", "chưa phát hiện",
    "không còn", "loại trừ", "âm tính", "phủ nhận", "không bị",
]
# Cue tiền sử — cue mạnh nhất.
HISTORICAL_CUES = [
    "tiền sử", "tiền căn", "trước khi nhập viện", "trước nhập viện",
    "đã được chẩn đoán",
]
# Cue người nhà — isFamily cực hiếm (4/15444). Chỉ cue rõ ràng "của người nhà",
# BỎ honorific ông/bà (dùng cho chính bệnh nhân) để tránh FP hàng loạt.
FAMILY_CUES = [
    "bố bệnh nhân", "mẹ bệnh nhân", "cha bệnh nhân", "gia đình bệnh nhân",
    "người nhà", "tiền sử gia đình", "bố đẻ", "mẹ đẻ", "anh trai", "chị gái",
]

# Ranh giới mệnh đề: xuống dòng, gạch đầu dòng, và dấu câu phân tách.
_CLAUSE_SPLIT = re.compile(r"[\n;.]|(?:^|\s)-\s|,\s")


def _clause_before(text: str, start: int) -> str:
    """Lấy phần mệnh đề TRƯỚC mention trong cùng câu/dòng (đã hạ chữ)."""
    ls = text.rfind("\n", 0, start) + 1
    left = text[ls:start]
    # cắt tại dấu phân tách mệnh đề gần nhất trước mention
    cuts = [m.end() for m in _CLAUSE_SPLIT.finditer(left)]
    if cuts:
        left = left[cuts[-1]:]
    return left.lower()


def _has_cue(clause: str, cues: list[str]) -> bool:
    for c in cues:
        if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", clause):
            return True
    return False


def infer_assertions(text: str, concept: Concept) -> list[str]:
    if concept.type not in TYPES_WITH_ASSERTIONS:
        return []
    start = concept.position[0]
    clause = _clause_before(text, start)
    mention = concept.text.lower().strip()

    out: list[str] = []
    # isNegated: cue phủ định trong mệnh đề trước mention, hoặc mention tự phủ định.
    if _has_cue(clause, NEGATION_CUES) or mention.startswith(("không", "chưa")):
        out.append("isNegated")
    # isHistorical: cue tiền sử trong mệnh đề trước mention.
    if _has_cue(clause, HISTORICAL_CUES):
        out.append("isHistorical")
    # isFamily: cue người nhà (cứng).
    if _has_cue(clause, FAMILY_CUES):
        out.append("isFamily")

    return out[:3]


def annotate(text: str, concepts: list[Concept]) -> list[Concept]:
    for c in concepts:
        c.assertions = infer_assertions(text, c)
    return concepts
