"""Dựng INPUT ngữ cảnh dùng CHUNG cho assertion-encoder và gate-encoder.

Trước đây mỗi module tự ghép `f"{type} | {context_window(...)}"`, và
`context_window` mặc định KHÔNG ra khỏi dòng chứa mention. Với layout của tập test
(mention chiếm trọn dòng gạch đầu dòng) input trở thành gần như chỉ có mention —
không còn ngữ cảnh nào để suy ra assertion hay "có được gán mã không".

Bản này bổ sung hai thứ:
1. **section path** — chuỗi heading bao ngoài ("Tiền sử bệnh › Các bệnh lý mạn tính").
   Đo được: chỉ 32% bullet trong test có cue tiền sử trong ±160 ký tự, nên cue thật
   nằm ở heading, thường cách xa hàng trăm ký tự.
2. **ngữ cảnh vượt dòng** — lấy thêm vài dòng trước/sau.

Cả hai đều là FEATURE, không phải rule: ground truth khá nhiễu (cùng "lý do vào viện"
có file gán `isHistorical`, file khác gán `[]`), nên để model tự học trọng số.

Dùng `build_inputs` khi xử lý nhiều concept trong cùng văn bản — nó tính `find_headings`
MỘT LẦN thay vì mỗi concept một lần.
"""
from __future__ import annotations

from ..assertion.llm import context_window
from ..io.schema import Concept
from .layout import Heading, find_headings, section_path

# Cấu hình mặc định — notebook có thể ghi đè để A/B.
DEFAULTS = dict(before=200, after=80, lines_before=2, lines_after=1,
                section=True, max_parts=2)


def build_input(text: str, concept: Concept, headings: list[Heading] | None = None,
                **kw) -> str:
    """`"{type} | {section path} | {ngữ cảnh có «mention»}"`."""
    cfg = {**DEFAULTS, **kw}
    ctx = context_window(text, concept, before=cfg["before"], after=cfg["after"],
                         lines_before=cfg["lines_before"],
                         lines_after=cfg["lines_after"])
    if not cfg["section"]:
        return f"{concept.type} | {ctx}"
    sec = section_path(text, concept.position[0], headings=headings,
                       max_parts=cfg["max_parts"])
    return f"{concept.type} | {sec} | {ctx}"


def build_inputs(text: str, concepts: list[Concept], **kw) -> list[str]:
    """Như `build_input` nhưng chỉ phân tích heading một lần cho cả văn bản."""
    cfg = {**DEFAULTS, **kw}
    hs = find_headings(text) if cfg["section"] else None
    return [build_input(text, c, headings=hs, **kw) for c in concepts]
