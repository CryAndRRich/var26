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
    """Chọn threshold TỪNG NHÃN để tối đa F1 của nhãn đó trên dev.

    probs/gold: ma trận (N, 3). Vì đoán THỪA assertion làm mất điểm Jaccard của
    concept đó, threshold thường cần > 0.5 (precision-first) — để dữ liệu tự quyết.
    Nhãn không có ca dương nào trong dev -> giữ 0.5 (không tune được).
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
