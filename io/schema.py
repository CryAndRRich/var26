"""Kiểu dữ liệu & I/O cho khái niệm y tế (input .txt / output .json).

Không hard-code path tuyệt đối — mọi hàm nhận path do caller truyền vào.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---- Enums (khớp TUYỆT ĐỐI với ground truth, kể cả dấu) ----
TYPES = (
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
)
ASSERTIONS = ("isNegated", "isFamily", "isHistorical")

# Loại nào được phép có candidates / assertions
TYPES_WITH_CANDIDATES = {"CHẨN_ĐOÁN", "THUỐC"}
TYPES_WITH_ASSERTIONS = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


@dataclass
class Concept:
    text: str
    type: str
    position: tuple[int, int]
    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Xuất dict đúng format nộp bài. Chỉ kèm field áp dụng cho type."""
        d: dict = {
            "text": self.text,
            "type": self.type,
            "position": [int(self.position[0]), int(self.position[1])],
        }
        if self.type in TYPES_WITH_CANDIDATES:
            d["candidates"] = list(self.candidates)
        if self.type in TYPES_WITH_ASSERTIONS:
            d["assertions"] = list(self.assertions)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        pos = d.get("position") or [0, 0]
        return cls(
            text=d.get("text", ""),
            type=d.get("type", ""),
            position=(int(pos[0]), int(pos[1])),
            assertions=list(d.get("assertions", []) or []),
            candidates=list(d.get("candidates", []) or []),
        )


# ---- I/O helpers ----
def read_input_text(path: str | Path) -> str:
    """Đọc file input GIỮ NGUYÊN ký tự gốc (không normalize) để offset khớp."""
    return Path(path).read_text(encoding="utf-8")


def load_concepts(json_path: str | Path) -> list[Concept]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return [Concept.from_dict(x) for x in data]


def save_concepts(concepts: Iterable[Concept], json_path: str | Path) -> None:
    out = [c.to_dict() for c in concepts]
    Path(json_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_concept(c: Concept, source_text: str) -> list[str]:
    """Trả list lỗi (rỗng nếu hợp lệ). Dùng để kiểm tra trước khi nộp."""
    errs: list[str] = []
    if c.type not in TYPES:
        errs.append(f"type không hợp lệ: {c.type!r}")
    s, e = c.position
    if not (0 <= s <= e <= len(source_text)):
        errs.append(f"position ngoài phạm vi: {c.position}")
    elif source_text[s:e] != c.text:
        errs.append(f"text != input[{s}:{e}]: {c.text!r} vs {source_text[s:e]!r}")
    if any(a not in ASSERTIONS for a in c.assertions):
        errs.append(f"assertion không hợp lệ: {c.assertions}")
    if len(c.assertions) > 3:
        errs.append("quá 3 assertions")
    if c.candidates and c.type not in TYPES_WITH_CANDIDATES:
        errs.append(f"type {c.type} không được có candidates")
    return errs
