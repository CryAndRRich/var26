"""Assertion bằng ENCODER multi-label classification (EXP3b) — phương án nhanh.

Bài toán assertion thực chất là phân loại 3 nhãn NHỊ PHÂN ĐỘC LẬP
(isNegated / isFamily / isHistorical) trên (ngữ cảnh + mention) → không cần LLM sinh
văn bản. Encoder train ~5-10' (vs 1-2h QLoRA 7B) và inference nhanh hơn hàng chục lần
— quan trọng khi submit: 100 file × ~154 concept ≈ 15.4k lần gán.

Dùng CHUNG cách đánh dấu mention («») với `assertion/llm.py::context_window` để hai
phương án so sánh công bằng trên cùng input.

Phần ở đây CPU-safe (chỉ tokenizer/numpy); vòng train đặt ở notebooks/ (Kaggle GPU).

LƯU Ý DỮ LIỆU: isFamily chỉ có 4 ca / 15.444 concept → gần như KHÔNG học được bằng
model. `EncoderAsserter(rule_labels=('isFamily',))` cho phép lấy nhãn đó từ rule và
để model lo 2 nhãn còn lại (hybrid).
"""
from __future__ import annotations

from ..io.schema import ASSERTIONS, Concept, TYPES_WITH_ASSERTIONS
from .llm import context_window

# Thứ tự nhãn CỐ ĐỊNH = thứ tự cột logits/threshold. Không đổi thứ tự này.
LABELS: tuple[str, ...] = ASSERTIONS
LABEL2IDX = {a: i for i, a in enumerate(LABELS)}


def build_input(text: str, concept: Concept) -> str:
    """Input cho classifier: kèm type (giúp phân biệt ngữ cảnh) + ngữ cảnh đã đánh dấu «»."""
    return f"{concept.type} | {context_window(text, concept)}"


def multi_hot(concept: Concept) -> list[float]:
    got = set(concept.assertions)
    return [1.0 if a in got else 0.0 for a in LABELS]


def build_examples(labeled: list[tuple[str, list[Concept]]]) -> list[dict]:
    """(text, concepts) -> [{'text', 'labels'(3 float), 'type'}] cho concept có assertions."""
    out: list[dict] = []
    for text, concepts in labeled:
        for c in concepts:
            if c.type not in TYPES_WITH_ASSERTIONS:
                continue
            out.append({"text": build_input(text, c), "labels": multi_hot(c), "type": c.type})
    return out


def featurize(examples: list[dict], tok, max_length: int = 160) -> list[dict]:
    """Tokenize -> feature cho HF Trainer (multi-label: labels là vector float)."""
    feats: list[dict] = []
    for e in examples:
        enc = tok(e["text"], truncation=True, max_length=max_length)
        feats.append({"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"],
                      "labels": e["labels"]})
    return feats


def tune_thresholds(probs, gold, grid=None) -> list[float]:
    """[KHÔNG DÙNG ĐỂ CHỌN NGƯỠNG CUỐI] Threshold tối đa F1 TỪNG NHÃN.

    ⚠️ Đo thực tế (EXP3b) cho thấy tối đa F1 làm ĐIỂM THI GIẢM: F1 đẩy ngưỡng xuống
    (0.15/0.40) để lấy recall, nhưng metric là **Jaccard theo concept** — đoán thừa 1
    assertion trên concept đáng-lẽ-rỗng làm sample đó 0 điểm. Kết quả: value 0.568 (tuned)
    < 0.589 (@0.5). Dùng `tune_thresholds_by_score` để tối ưu ĐÚNG hàm mục tiêu.
    Giữ hàm này chỉ để phân tích P/R/F1 từng nhãn.

    probs/gold: ma trận (N, 3). Nhãn không có ca dương trong dev -> giữ 0.5.
    """
    grid = grid if grid is not None else [i / 20 for i in range(2, 19)]  # 0.10..0.90
    out: list[float] = []
    for j in range(len(LABELS)):
        best_t, best_f1 = 0.5, -1.0
        n_pos = sum(1 for i in range(len(gold)) if gold[i][j] > 0.5)
        if n_pos == 0:
            out.append(0.5)
            continue
        for t in grid:
            tp = fp = fn = 0
            for i in range(len(gold)):
                p, g = probs[i][j] >= t, gold[i][j] > 0.5
                tp += p and g
                fp += p and not g
                fn += (not p) and g
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        out.append(best_t)
    return out


def tune_thresholds_by_score(dev_labeled, asserter, key_mode: str = "value",
                             grid=None, passes: int = 4, start=None, verbose: bool = True):
    """Chọn threshold để tối đa CHÍNH `assertions_score` (Jaccard) trên dev.

    Đây là cách đúng: metric phạt nặng đoán thừa (concept đáng-lẽ-rỗng mà gán nhãn -> 0
    điểm sample đó), nên tối ưu F1 từng nhãn cho kết quả tệ hơn (xem `tune_thresholds`).

    Dùng coordinate ascent (quét lần lượt từng nhãn, lặp `passes` lượt) — rẻ hơn quét
    toàn bộ tổ hợp mà thực tế đủ tốt. Xác suất được TÍNH MỘT LẦN rồi tái dùng cho mọi
    ngưỡng nên vòng tune không cần chạy lại model.

    dev_labeled: [(text, gold_concepts)] — thường là DEV, KHÔNG phải tập báo cáo cuối.
    Trả (thresholds, best_score).
    """
    import copy

    # Chỉ cần THÀNH PHẦN assertion — gọi score_dataset đầy đủ sẽ tính cả text score (WER,
    # O(n²) mỗi file) khiến vòng tune chậm gấp hàng chục lần một cách vô ích.
    from ..eval.metrics import assertions_score_sample

    grid = grid if grid is not None else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # 1) tính prob 1 lần cho toàn dev
    cached = []
    for text, gold in dev_labeled:
        todo = [c for c in gold if c.type in TYPES_WITH_ASSERTIONS]
        probs = asserter.predict_proba([build_input(text, c) for c in todo])
        cached.append((gold, todo, probs))

    id_of = {}   # id(concept) -> chỉ số trong todo, để gán nhanh
    for gold, todo, _ in cached:
        for k, c in enumerate(todo):
            id_of[id(c)] = k

    def evaluate(th: list[float]) -> float:
        total = 0.0
        for gold, todo, probs in cached:
            pred = []
            for c in gold:
                c2 = copy.copy(c)
                if c.type in TYPES_WITH_ASSERTIONS:
                    p = probs[id_of[id(c)]]
                    c2.assertions = [a for j, a in enumerate(LABELS) if p[j] >= th[j]]
                else:
                    c2.assertions = []
                pred.append(c2)
            total += assertions_score_sample(gold, pred, key_mode)
        return total / len(cached) if cached else 0.0

    th = list(start) if start is not None else [0.5] * len(LABELS)
    best = evaluate(th)
    if verbose:
        print(f"  khởi đầu th={th} -> {key_mode} assert={best:.4f}")
    for p in range(passes):
        improved = False
        for j in range(len(LABELS)):
            for t in grid:
                if t == th[j]:
                    continue
                cand = list(th)
                cand[j] = t
                s = evaluate(cand)
                if s > best + 1e-9:
                    best, th, improved = s, cand, True
        if verbose:
            print(f"  pass {p + 1}: th={th} -> {best:.4f}")
        if not improved:
            break
    return th, best


class EncoderAsserter:
    """Bọc model sequence-classification multi-label. Interface .annotate như rules/LLMAsserter.

    thresholds: 3 ngưỡng theo thứ tự LABELS (mặc định 0.5 hết).
    rule_labels: nhãn LẤY TỪ RULE thay vì model (vd ('isFamily',) vì chỉ 4 ca trong train).
    """

    def __init__(self, model, tokenizer, thresholds=None, batch_size: int = 64,
                 max_length: int = 160, rule_labels: tuple[str, ...] = ()):
        self.model = model
        self.tokenizer = tokenizer
        self.thresholds = list(thresholds) if thresholds is not None else [0.5] * len(LABELS)
        self.batch_size = batch_size
        self.max_length = max_length
        self.rule_labels = tuple(rule_labels)
        self.model.eval()

    def predict_proba(self, inputs: list[str]):
        """list text -> list[list[float]] xác suất sigmoid (N, 3)."""
        import torch

        if not inputs:
            return []
        device = getattr(self.model, "device", None) or "cpu"
        out: list[list[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            chunk = inputs[i:i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.extend(torch.sigmoid(logits).float().cpu().tolist())
        return out

    def annotate(self, text: str, concepts: list[Concept]) -> list[Concept]:
        from .rules import infer_assertions as rule_infer

        todo = [c for c in concepts if c.type in TYPES_WITH_ASSERTIONS]
        for c in concepts:
            if c.type not in TYPES_WITH_ASSERTIONS:
                c.assertions = []

        probs = self.predict_proba([build_input(text, c) for c in todo])
        for c, p in zip(todo, probs):
            got = [a for j, a in enumerate(LABELS)
                   if a not in self.rule_labels and p[j] >= self.thresholds[j]]
            if self.rule_labels:  # bổ sung nhãn hiếm bằng rule
                r = set(rule_infer(text, c))
                got += [a for a in self.rule_labels if a in r]
            c.assertions = [a for a in ASSERTIONS if a in set(got)]  # thứ tự chuẩn
        return concepts
