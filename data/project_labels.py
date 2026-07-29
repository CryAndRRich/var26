"""Chiếu nhãn gold từ tập CÓ NHÃN sang tập KHÔNG NHÃN qua đoạn trùng khớp nguyên văn.

## Vì sao dùng được

Cả `data/train` và `data/test` đều được dựng bằng cách GHÉP nhiều đoạn tư liệu. Đo
được: 12 file test có 10–76% ký tự trùng nguyên văn với một file train nào đó (test/24:
76%, test/36: 73%, test/40: 54%, test/53: 49%). Với những vùng trùng đó ta biết CHÍNH
XÁC nhãn của ban tổ chức, đặt trong ngữ cảnh file test thật.

⟹ Có được **dev đo trên chính tập test** thay vì chỉ đo trên train. Đây là tập dev duy
nhất có nhãn thật trên test; dùng nó để bắt lỗi hồi quy, KHÔNG dùng để tune sâu (nhỏ,
và các vùng trùng đều là văn xuôi kiểu train nên KHÔNG đại diện cho layout gạch đầu
dòng của test — phần đó dùng dev tổng hợp ở `layout_synth`).

## Kỷ luật bắt buộc: LEAVE-SOURCE-OUT
Nhãn chiếu sang test bắt nguồn từ file train nào thì file train ĐÓ phải bị loại khỏi
tập train của model, nếu không điểm đo sẽ là điểm học thuộc. `sources_of` trả danh
sách đó.

Thuần CPU, không phụ thuộc gì ngoài stdlib.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..io.schema import Concept

SHINGLE = 32   # độ dài mỏ neo; đủ dài để không khớp ngẫu nhiên trong tiếng Việt


def _index(sources: dict[str, str], k: int) -> dict[str, list[tuple[str, int]]]:
    idx: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sid, t in sources.items():
        for i in range(len(t) - k + 1):
            idx[t[i:i + k]].append((sid, i))
    return idx


def build_index(sources: dict[str, str], k: int = SHINGLE):
    """Chỉ mục mỏ neo. Dựng MỘT LẦN rồi dùng lại cho mọi target (nếu không, 100 target
    x ~1M shingle mất ~110s thay vì ~10s)."""
    return _index(sources, k)


def align(target: str, sources: dict[str, str], k: int = SHINGLE,
          idx=None) -> dict[int, tuple[str, int]]:
    """{vị trí trong target: (source_id, vị trí trong source)} cho các đoạn khớp dài."""
    idx = _index(sources, k) if idx is None else idx
    out: dict[int, tuple[str, int]] = {}
    i = 0
    while i <= len(target) - k:
        cands = idx.get(target[i:i + k])
        if not cands:
            i += 1
            continue
        best = None
        for sid, sp in cands:                      # mở rộng tối đa về bên phải
            src = sources[sid]
            n = k
            while i + n < len(target) and sp + n < len(src) and target[i + n] == src[sp + n]:
                n += 1
            if best is None or n > best[2]:
                best = (sid, sp, n)
        sid, sp, n = best
        for j in range(n):
            out.setdefault(i + j, (sid, sp + j))
        i += n
    return out


def project(target: str, sources: dict[str, str],
            gold: dict[str, list[Concept]], k: int = SHINGLE, idx=None
            ) -> tuple[list[Concept], list[tuple[int, int]], Counter]:
    """-> (concept đã chiếu, các khoảng ký tự đã phủ, đếm ký tự theo source).

    Một concept chỉ được chiếu khi TOÀN BỘ span của nó nằm trong vùng khớp, ánh xạ
    thành một dải LIÊN TỤC trong target, và `target[s:e]` khớp đúng `concept.text`.
    """
    fwd = align(target, sources, k, idx=idx)
    rev = {v: t for t, v in fwd.items()}
    srcs = Counter(sid for sid, _ in fwd.values())

    proj: list[Concept] = []
    for sid in srcs:
        for c in gold.get(sid, []):
            s, e = c.position
            tps = [rev.get((sid, p)) for p in range(s, e)]
            if not tps or any(x is None for x in tps):
                continue
            if tps != list(range(tps[0], tps[0] + len(tps))):
                continue
            if target[tps[0]:tps[-1] + 1] != c.text:
                continue
            proj.append(Concept(text=c.text, type=c.type,
                                position=(tps[0], tps[-1] + 1),
                                assertions=list(c.assertions),
                                candidates=list(c.candidates)))
    seen, uniq = set(), []
    for c in sorted(proj, key=lambda x: (x.position, x.type)):
        key = (c.position[0], c.position[1], c.type)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    covered = sorted(fwd)
    spans: list[tuple[int, int]] = []
    for p in covered:
        if spans and p == spans[-1][1]:
            spans[-1] = (spans[-1][0], p + 1)
        else:
            spans.append((p, p + 1))
    return uniq, spans, srcs


def build_dev(targets: dict[str, str], sources: dict[str, str],
              gold: dict[str, list[Concept]], min_cov: float = 0.15,
              min_concepts: int = 4, k: int = SHINGLE,
              max_files: int | None = None) -> list[dict]:
    """Dựng dev chiếu-nhãn cho mọi target đạt ngưỡng phủ.

    Mỗi phần tử: {'id', 'text', 'concepts', 'spans', 'coverage', 'sources'}.
    `spans` = vùng có nhãn tin được; khi chấm điểm PHẢI lọc cả gold và prediction về
    trong vùng này, nếu không phần chưa phủ sẽ bị tính oan thành dự đoán sai.
    """
    idx = build_index(sources, k)
    out = []
    for tid, text in targets.items():
        concepts, spans, srcs = project(text, sources, gold, k=k, idx=idx)
        cov = sum(e - s for s, e in spans) / max(1, len(text))
        if cov < min_cov or len(concepts) < min_concepts:
            continue
        out.append({"id": tid, "text": text, "concepts": concepts, "spans": spans,
                    "coverage": cov, "sources": sorted(srcs)})
    out.sort(key=lambda d: -d["coverage"])
    return out[:max_files] if max_files else out


def sources_of(dev: list[dict]) -> set[str]:
    """Các file train ĐÃ đóng góp nhãn -> phải loại khỏi tập train (leave-source-out)."""
    return {s for d in dev for s in d["sources"]}


def clip_to_spans(concepts: list[Concept], spans: list[tuple[int, int]]) -> list[Concept]:
    """Giữ concept nằm HẲN trong vùng có nhãn tin được."""
    return [c for c in concepts
            if any(s <= c.position[0] and c.position[1] <= e for s, e in spans)]
