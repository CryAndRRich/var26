"""Bộ chấm điểm cục bộ mô phỏng metric của đề (BƯỚC 3: bản proxy).

final = 0.3*text + 0.3*assertions + 0.4*candidates

⚠️ LƯU Ý QUAN TRỌNG: Đề mô tả công thức nhưng KHÔNG công bố mã chấm chính thức.
Một số điểm còn nhập nhằng (cách gộp set assertions/candidates ở cấp sample,
cách tính WER trên trường text). File này cài đặt một bản DIỄN GIẢI HỢP LÝ,
tách module để dễ thay khi có scorer chính thức. Dùng để so sánh TƯƠNG ĐỐI
giữa các phiên bản model, không coi là điểm tuyệt đối.

Giả định đã ghi rõ ở từng hàm.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..io.schema import Concept, TYPES_WITH_ASSERTIONS, TYPES_WITH_CANDIDATES


# --------------------------------------------------------------------------
# Word Error Rate (word-level Levenshtein)
# --------------------------------------------------------------------------
def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """WER giữa 2 chuỗi (word-level). Có thể > 1; caller tự cap khi cần."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def text_score_sample(gt: list[Concept], pred: list[Concept]) -> float:
    """1 - WER trên trường `text` của sample.

    GIẢ ĐỊNH: nối các `text` (theo thứ tự xuất hiện) thành 1 chuỗi từ,
    tính WER giữa GT và prediction. (1 - WER) cap tại 0.
    """
    ref = " ".join(c.text for c in gt)
    hyp = " ".join(c.text for c in pred)
    return max(0.0, 1.0 - wer(ref, hyp))


# --------------------------------------------------------------------------
# Jaccard cấp sample (dùng cho assertions & candidates)
# --------------------------------------------------------------------------
def _jaccard(gt_items: Counter, pred_items: Counter) -> float:
    """Jaccard trên multiset. Theo đề:
    - GT rỗng & pred rỗng -> 1
    - GT rỗng & pred khác rỗng -> 0
    - còn lại -> |giao| / |hợp|
    """
    if not gt_items and not pred_items:
        return 1.0
    if not gt_items:
        return 0.0
    inter = sum((gt_items & pred_items).values())
    union = sum((gt_items | pred_items).values())
    return inter / union if union else 1.0


def _align_key(c: Concept) -> tuple:
    """Khóa gióng concept GT<->pred. Sai type bị tính là concept khác (đề: phạt kép)."""
    return (c.position[0], c.position[1], c.type)


# key_mode: cách định nghĩa "tập" cấp sample (mã chấm chính thức chưa công bố):
#   "value"   (A): tập GIÁ TRỊ thô (code/assertion) — đọc theo nghĩa đen "set" của đề.
#   "concept" (B): tập (concept_key, giá trị) — gắn giá trị với đúng khái niệm.
#
# BẰNG CHỨNG TỪ ĐỀ nghiêng về (B): mục 5a ghi "đoán đúng phần `text` nhưng sai `type` thì
# khái niệm bị TÍNH 2 LẦN và mỗi lần đều 0 điểm với cả 3 metric". Câu này chỉ có nghĩa nếu
# scorer GIÓNG prediction với gold theo từng khái niệm — tức là (B). Với (A) (túi giá trị
# cấp file) thì sai `type` không tạo ra hiện tượng "tính 2 lần" nào.
# ⟹ Khi chọn ngưỡng, ưu tiên "concept"; vẫn báo cáo cả hai vì chưa có scorer chính thức.
# Xem thêm docs/results/2026-07-29_domain_shift.md.
CONCEPT_FIRST_W = {"concept": 2 / 3, "value": 1 / 3}


def _item_counter(concepts, types, field, key_mode):
    out = Counter()
    for c in concepts:
        if c.type not in types:
            continue
        for v in getattr(c, field):
            out[v if key_mode == "value" else (_align_key(c), v)] += 1
    return out


def assertions_score_sample(gt, pred, key_mode: str = "value") -> float:
    g = _item_counter(gt, TYPES_WITH_ASSERTIONS, "assertions", key_mode)
    p = _item_counter(pred, TYPES_WITH_ASSERTIONS, "assertions", key_mode)
    return _jaccard(g, p)


def candidates_score_sample(gt, pred, key_mode: str = "value") -> tuple[float, int]:
    g = _item_counter(gt, TYPES_WITH_CANDIDATES, "candidates", key_mode)
    p = _item_counter(pred, TYPES_WITH_CANDIDATES, "candidates", key_mode)
    weight = sum(
        len(c.candidates) + 1 for c in gt if c.type in TYPES_WITH_CANDIDATES
    )
    return _jaccard(g, p), weight


# --------------------------------------------------------------------------
# Tổng hợp
# --------------------------------------------------------------------------
@dataclass
class Scores:
    text: float
    assertions: float
    candidates: float

    @property
    def final(self) -> float:
        return 0.3 * self.text + 0.3 * self.assertions + 0.4 * self.candidates


def score_dataset(
    gts: list[list[Concept]],
    preds: list[list[Concept]],
    key_mode: str = "value",
) -> Scores:
    """Chấm điểm toàn tập. gts[i], preds[i] là list concept của sample i.

    key_mode: "value" (mặc định, đọc nghĩa đen) hoặc "concept" (gắn khái niệm).
    """
    assert len(gts) == len(preds), "số sample GT và pred phải bằng nhau"
    n = len(gts)
    if n == 0:
        return Scores(0.0, 0.0, 0.0)

    text_sum = sum(text_score_sample(g, p) for g, p in zip(gts, preds))
    assert_sum = sum(
        assertions_score_sample(g, p, key_mode) for g, p in zip(gts, preds)
    )

    cand_num = 0.0
    cand_den = 0.0
    for g, p in zip(gts, preds):
        j, w = candidates_score_sample(g, p, key_mode)
        cand_num += j * w
        cand_den += w
    cand = cand_num / cand_den if cand_den else 1.0

    return Scores(text_sum / n, assert_sum / n, cand)
