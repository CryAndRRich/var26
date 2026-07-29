"""Phân tích BỐ CỤC văn bản (heading / bullet / section path).

## Vì sao cần file này

Đo thực tế (2026-07-29) trên `data/`:

| | train (100) | test (100) |
|---|---|---|
| bullet `- …` mỗi file | 2.0 | **12.8** |
| % ký tự nằm trong dòng bullet | 0.8% | **31.7%** |
| file có ≥10 bullet | **0** | **52** |
| `Bệnh nhân là` | 90 file | **0** |
| `Câu hỏi từ người dùng` | **0** | 35 file |
| `Lý do nhập viện` | **0** | 49 file |
| trùng 8-gram với train | — | **4.2%** |

⟹ TRAIN là văn xuôi bệnh án dịch (MIMIC-like) bọc trong "PHIẾU BÀN GIAO/BÁO CÁO";
TEST là **bản tóm tắt có cấu trúc theo mục + gạch đầu dòng**, hỏi-đáp web, bài phổ biến
kiến thức. Gần 1/3 ký tự của test nằm trong layout mà train hầu như KHÔNG có.

Hệ quả trực tiếp đo được: chỉ **32%** bullet trong test có cue tiền sử ("tiền sử",
"bệnh lý mạn tính", "thuốc trước khi nhập viện"…) nằm trong cửa sổ ±160 ký tự mà
`assertion.llm.context_window` cấp cho model — tức 68% lần model **mù** với cue quyết
định. Trong train tỉ lệ này là 41% (và văn xuôi còn cue tại chỗ "có tiền sử X").

⟹ Cách sửa: coi **section path** (chuỗi heading bao ngoài) là FEATURE cấp cho model,
không phải rule cứng (ground truth khá nhiễu: cùng "lý do vào viện" có file gán
`isHistorical`, file khác gán `[]`).

Toàn bộ file CPU-safe, không phụ thuộc dữ liệu ngoài repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Dòng bắt đầu bằng dấu gạch/bullet -> KHÔNG phải heading, là item.
# CHỈ nhận `- – — •` (đúng những gì tập test dùng). KHÔNG nhận `*`: trong train có dòng
# phân cách `***` ngay dưới "Độc lập - Tự do - Hạnh phúc", làm dòng đó bị coi là heading
# và section path của cả vùng thành rác.
_BULLET = re.compile(r"^[ \t]*([-–—•]|\d+[\.\)])[ \t]+")
_BULLET_ANY = re.compile(r"^[ \t]*[-–—•][ \t]*")
# Heading có số thứ tự: "1. Tiền sử bệnh", "2)  Bệnh sử"
_NUMBERED = re.compile(r"^[ \t]*(\d+)[\.\)][ \t]*(.+?)[ \t]*$")

MAX_HEADING_LEN = 70


@dataclass
class Heading:
    start: int          # offset ký tự đầu dòng heading trong text gốc
    end: int            # offset cuối dòng
    level: int          # 1 = mục có số, 2 = heading phụ
    title: str          # nội dung heading đã strip (bỏ số thứ tự, bỏ ':')


def _is_bullet_item(line: str) -> bool:
    return bool(_BULLET_ANY.match(line))


def _clean_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\d+[\.\)]\s*", "", s)
    return s.strip().rstrip(":").strip()


def find_headings(text: str) -> list[Heading]:
    """Tìm các dòng heading. Heuristic bám đúng layout của tập test.

    Một dòng là heading nếu: NGẮN (<= MAX_HEADING_LEN), không phải bullet item, và
      (a) có số thứ tự dạng "1." / "2)"  -> level 1, hoặc
      (b) kết thúc bằng ':'              -> level 2, hoặc
      (c) dòng kế tiếp (không rỗng) là bullet item hoặc thụt lề sâu hơn -> level 2.

    (c) là điều kiện quan trọng nhất với test: "Các bệnh lý mạn tính" không có ':'
    nhưng ngay dưới là danh sách gạch đầu dòng.
    """
    lines: list[tuple[int, int, str]] = []
    pos = 0
    for ln in text.split("\n"):
        lines.append((pos, pos + len(ln), ln))
        pos += len(ln) + 1

    def next_nonempty(i: int) -> str | None:
        for j in range(i + 1, len(lines)):
            if lines[j][2].strip():
                return lines[j][2]
        return None

    out: list[Heading] = []
    for i, (s, e, ln) in enumerate(lines):
        body = ln.strip()
        if not body or len(body) > MAX_HEADING_LEN:
            continue
        if _is_bullet_item(ln):
            continue
        if body.endswith((".", "?", "!")) and not body.endswith(":"):
            # câu hoàn chỉnh -> không coi là heading (trừ khi có số thứ tự, xử lý dưới)
            if not _NUMBERED.match(ln):
                continue
        m = _NUMBERED.match(ln)
        if m:
            title = _clean_title(m.group(2))
            if title:
                out.append(Heading(s, e, 1, title))
            continue
        if body.endswith(":"):
            title = _clean_title(body)
            if title:
                out.append(Heading(s, e, 2, title))
            continue
        nxt = next_nonempty(i)
        if nxt is not None and (_is_bullet_item(nxt) or
                               (len(nxt) - len(nxt.lstrip()) > len(ln) - len(ln.lstrip()))):
            title = _clean_title(body)
            if title:
                out.append(Heading(s, e, 2, title))
    return out


def section_path(text: str, pos: int, headings: list[Heading] | None = None,
                 max_parts: int = 2) -> str:
    """Chuỗi heading bao ngoài vị trí `pos`, gần nhất đứng cuối.

    Ví dụ: "Tiền sử bệnh › Các bệnh lý mạn tính".
    Trả "" nếu không có heading nào trước `pos`.
    """
    hs = find_headings(text) if headings is None else headings
    stack: dict[int, str] = {}
    order: list[int] = []
    for h in hs:
        if h.start >= pos:
            break
        # heading level L xoá mọi level >= L
        for lv in [lv for lv in stack if lv >= h.level]:
            del stack[lv]
        stack[h.level] = h.title
        order = sorted(stack)
    parts = [stack[lv] for lv in order]
    if max_parts > 0:
        parts = parts[-max_parts:]
    return " › ".join(parts)


def line_span(text: str, pos: int) -> tuple[int, int]:
    """[đầu dòng, cuối dòng) chứa `pos`."""
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e < 0 else e)


def in_bullet_line(text: str, pos: int) -> bool:
    s, e = line_span(text, pos)
    return _is_bullet_item(text[s:e])


def layout_stats(text: str) -> dict:
    """Thống kê layout 1 văn bản — dùng để phát hiện lệch phân phối train/test."""
    lines = text.split("\n")
    bl = [l for l in lines if _BULLET_ANY.match(l)]
    return {
        "n_lines": len(lines),
        "n_bullets": len(bl),
        "bullet_char_ratio": sum(len(l) for l in bl) / max(1, len(text)),
        "n_headings": len(find_headings(text)),
    }
