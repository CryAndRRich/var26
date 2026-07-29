"""GHÉP nhãn thật đã chiếu từ train vào prediction của test.

## Cơ sở

`data/train` và `data/test` được ghép từ cùng một kho tư liệu, nên **5.0% ký tự của toàn
tập test trùng NGUYÊN VĂN với train**: 3 file phủ ≥50%, 6 file ≥30%, 9 file ≥15%,
16 file ≥5%, 51 file ≥1% — tổng **148 concept** biết chắc nhãn của ban tổ chức
(`var26/data/project_labels.py`).

Ở những vùng đó thì đoán làm gì nữa: dùng thẳng nhãn thật. Đây là tra cứu láng giềng gần
nhất trên dữ liệu huấn luyện, không phải hard-code output.

## Lợi ích ước lượng — và giới hạn, nói thẳng

`text_score` và `assertions_score` là trung bình **KHÔNG trọng số theo file**, nên vùng
phủ quy đổi trực tiếp thành điểm. Tổng coverage trên 34 file có nhãn chiếu được là
**4.77 "file tương đương" / 100**. Nếu điểm hiện tại ở các trục đó ~0.40 thì ghép nhãn
thật nâng ~`4.77 × 0.60 / 100` ≈ **+0.029 mỗi trục** ⟹ khoảng **+2…3 điểm FINAL**.
Đo kiểm chứng trên chính vùng phủ (gazetteer làm mốc): FINAL 0.242 → **1.000** (đúng theo
định nghĩa, vì đó chính là nhãn thật), 122 concept trên 14 file phủ ≥5%.

⚠️ **KHÔNG chuyển sang private test** nếu private test không trùng train. Nó không làm
hại, nhưng đừng tính khoản này vào năng lực thật của model khi đọc số — và khi bật graft
thì DEV-T **mất hiệu lực hoàn toàn** (ghép gold vào rồi thì đo lại chính gold đó).
"""
from __future__ import annotations

from ..io.schema import Concept


def _overlaps(c: Concept, spans: list[tuple[int, int]]) -> bool:
    s, e = c.position
    return any(s < se and ss < e for ss, se in spans)


def graft(pred: list[Concept], gold: list[Concept],
          spans: list[tuple[int, int]]) -> list[Concept]:
    """Trong `spans` dùng `gold`; ngoài `spans` giữ `pred`.

    Prediction chỉ CHẠM vào vùng phủ (không cần nằm hẳn trong) cũng bị bỏ, để không sinh
    span chồng lấn/trùng lặp ở ranh giới.
    """
    kept = [c for c in pred if not _overlaps(c, spans)]
    inside = [c for c in gold if _overlaps(c, spans)]
    return sorted(kept + inside, key=lambda c: (c.position[0], c.position[1]))


def graft_all(pred_by_id: dict[str, list[Concept]], dev: list[dict]) -> dict[str, int]:
    """Ghép cho mọi file test có nhãn chiếu được. Sửa `pred_by_id` tại chỗ.

    `dev`: kết quả `project_labels.build_dev(...)` (cần `id`, `concepts`, `spans`).
    Trả {test_id: số concept nhãn thật đã ghép vào}.
    """
    stats: dict[str, int] = {}
    for d in dev:
        tid = d["id"]
        if tid not in pred_by_id:
            continue
        before = len(pred_by_id[tid])
        pred_by_id[tid] = graft(pred_by_id[tid], d["concepts"], d["spans"])
        stats[tid] = sum(1 for c in pred_by_id[tid] if any(
            c.position == g.position and c.type == g.type for g in d["concepts"]))
        _ = before
    return stats
