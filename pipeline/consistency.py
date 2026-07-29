"""Bỏ phiếu NHẤT QUÁN giữa các file test gần trùng nhau.

## Cơ sở

**53/100 file test nằm trong 21 cụm gần trùng** (Jaccard trên 8-gram từ > 0.3; cụm lớn
nhất gồm 5 file: 35/56/67/86/94; các cụm khác: 14/19/28/52, 30/44/76/83, 13/16/20…).
Cùng một đoạn văn xuất hiện ở nhiều file với ngữ cảnh xung quanh hơi khác, nên model có
thể ra kết quả KHÁC NHAU trên cùng nội dung — thuần nhiễu.

Ghép các file trong cụm lại rồi bỏ phiếu = ensemble miễn phí trên chính nội dung đó, giảm
phương sai mà không cần chạy model thêm lần nào. Khác với `graft.py`, cách này KHÔNG dùng
nhãn thật nên vẫn có tác dụng trên private test (miễn là private test cũng có file trùng).

Chỉ bỏ phiếu TRONG vùng đã dóng được; ngoài vùng đó giữ nguyên prediction gốc — nếu không
sẽ xoá mất concept ở phần nội dung riêng của từng file.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..data.project_labels import align, build_index
from ..io.schema import Concept


def _shingles(text: str, k: int = 8) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def clusters(texts: dict[str, str], min_jaccard: float = 0.3,
             k: int = 8) -> list[list[str]]:
    """Nhóm các văn bản gần trùng (thành phần liên thông theo Jaccard 8-gram)."""
    sh = {i: _shingles(t, k) for i, t in texts.items()}
    ids = list(texts)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            un = len(sh[a] | sh[b])
            if un and len(sh[a] & sh[b]) / un >= min_jaccard:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def _transfer(src_text: str, src_concepts: list[Concept], dst_text: str
              ) -> tuple[list[Concept], list[tuple[int, int]]]:
    """Chuyển concept của `src` sang toạ độ của `dst` qua đoạn trùng nguyên văn.

    Trả (concept đã chuyển, các khoảng của dst được dóng với src).
    """
    fwd = align(dst_text, {"s": src_text}, idx=build_index({"s": src_text}))
    rev = {sp: dp for dp, (_sid, sp) in fwd.items()}
    out: list[Concept] = []
    for c in src_concepts:
        s, e = c.position
        dps = [rev.get(p) for p in range(s, e)]
        if not dps or any(x is None for x in dps):
            continue
        if dps != list(range(dps[0], dps[0] + len(dps))):
            continue
        if dst_text[dps[0]:dps[-1] + 1] != c.text:
            continue
        out.append(Concept(text=c.text, type=c.type,
                           position=(dps[0], dps[-1] + 1),
                           assertions=list(c.assertions),
                           candidates=list(c.candidates)))
    covered = sorted(fwd)
    spans: list[tuple[int, int]] = []
    for p in covered:
        if spans and p == spans[-1][1]:
            spans[-1] = (spans[-1][0], p + 1)
        else:
            spans.append((p, p + 1))
    return out, spans


def _in(c: Concept, spans: list[tuple[int, int]]) -> bool:
    s, e = c.position
    return any(ss <= s and e <= se for ss, se in spans)


def vote_cluster(texts: dict[str, str], preds: dict[str, list[Concept]],
                 group: list[str], keep_ratio: float = 0.5
                 ) -> dict[str, list[Concept]]:
    """Bỏ phiếu trong 1 cụm. Trả prediction MỚI cho từng file trong cụm.

    Với mỗi file `t`: mỗi file khác `o` trong cụm góp phiếu cho vùng mà `t` dóng được với
    `o`. Một span được giữ nếu được **≥ keep_ratio** số phiếu ĐỦ ĐIỀU KIỆN (tức chỉ tính
    những `o` mà vùng đó có mặt trong `o`). `assertions`/`candidates` lấy theo đa số.
    """
    out: dict[str, list[Concept]] = {}
    for t in group:
        others = [o for o in group if o != t]
        transferred, spans_of = {}, {}
        for o in others:
            transferred[o], spans_of[o] = _transfer(texts[o], preds[o], texts[t])

        # gom ứng viên: của chính t + chuyển từ các file khác
        cand: dict[tuple[int, int, str], list[Concept]] = defaultdict(list)
        for c in preds[t]:
            cand[(c.position[0], c.position[1], c.type)].append(c)
        for o in others:
            for c in transferred[o]:
                cand[(c.position[0], c.position[1], c.type)].append(c)

        keep: list[Concept] = []
        for (s, e, typ), members in cand.items():
            probe = Concept(text=texts[t][s:e], type=typ, position=(s, e))
            eligible = [o for o in others if _in(probe, spans_of[o])]
            if not eligible:                    # vùng riêng của t -> giữ nguyên
                keep.append(members[0])
                continue
            own = any(c.position == (s, e) and c.type == typ for c in preds[t])
            votes = (1 if own else 0) + sum(
                1 for o in eligible
                if any(c.position == (s, e) and c.type == typ for c in transferred[o]))
            if votes / (len(eligible) + 1) < keep_ratio:
                continue
            a = Counter(tuple(c.assertions) for c in members).most_common(1)[0][0]
            cd = Counter(tuple(c.candidates) for c in members).most_common(1)[0][0]
            keep.append(Concept(text=probe.text, type=typ, position=(s, e),
                                assertions=list(a), candidates=list(cd)))
        out[t] = sorted(keep, key=lambda c: (c.position[0], c.position[1]))
    return out


def vote_all(texts: dict[str, str], preds: dict[str, list[Concept]],
             min_jaccard: float = 0.3, keep_ratio: float = 0.5,
             verbose: bool = True) -> dict[str, list[Concept]]:
    """Bỏ phiếu cho MỌI cụm gần trùng. Trả bản prediction mới (không sửa `preds`)."""
    out = {k: list(v) for k, v in preds.items()}
    groups = clusters(texts, min_jaccard=min_jaccard)
    if verbose:
        n = sum(len(g) for g in groups)
        print(f"  {len(groups)} cụm gần trùng, phủ {n}/{len(texts)} file")
    for g in groups:
        before = sum(len(out[i]) for i in g)
        new = vote_cluster(texts, out, g, keep_ratio=keep_ratio)
        out.update(new)
        if verbose:
            after = sum(len(new[i]) for i in g)
            print(f"    {g}: {before} -> {after} concept")
    return out
