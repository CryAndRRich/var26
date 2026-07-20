"""Encoder token-classification NER (XLM-R/PhoBERT) — EXP1.

Chứa phần TÁI SỬ DỤNG & nhẹ (không train ở đây):
  - Sơ đồ nhãn BIO theo 5 type.
  - char spans <-> token labels (dùng offset_mapping của fast tokenizer).
  - EncoderTagger: inference (sliding window) -> list[Concept], offset khớp input RAW.

Vòng lặp train (HF Trainer) đặt ở notebooks/ (Kaggle GPU). torch/transformers chỉ
import khi thực sự dùng tagger, nên import module này ở máy local (không GPU) vẫn OK
tới các hàm BIO thuần Python.
"""
from __future__ import annotations

from ..io.schema import Concept, TYPES


# ------------------------- Sơ đồ nhãn BIO -------------------------
def build_labels() -> list[str]:
    labels = ["O"]
    for t in TYPES:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


LABELS = build_labels()
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
O_ID = LABEL2ID["O"]


# ------------------------- char spans -> token labels -------------------------
def align_labels(offsets, spans, ignore_index: int = -100) -> list[int]:
    """offsets: list (char_start, char_end) mỗi token (special token = (0,0)).
    spans: list (start, end, type). Trả list nhãn id theo BIO; special token = ignore_index.
    """
    labels = [ignore_index if cs == ce else O_ID for (cs, ce) in offsets]
    for (s, e, t) in sorted(spans, key=lambda x: x[0]):
        b_id, i_id = LABEL2ID[f"B-{t}"], LABEL2ID[f"I-{t}"]
        first = True
        for idx, (cs, ce) in enumerate(offsets):
            if cs == ce:
                continue
            if cs < e and ce > s:  # token giao với span
                labels[idx] = b_id if first else i_id
                first = False
    return labels


# ------------------------- token labels -> char spans -------------------------
def _clean_span(text: str, s: int, e: int) -> tuple[int, int]:
    """Bỏ khoảng trắng đầu/cuối để text[s:e] gọn (khớp cách gán nhãn của đề)."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def decode_spans(text: str, offsets, label_ids) -> list[Concept]:
    """Ghép run B/I liên tiếp cùng type -> Concept. offsets/label_ids theo thứ tự token."""
    out: list[Concept] = []
    cur_type = None
    cur_s = cur_e = None

    def flush():
        nonlocal cur_type, cur_s, cur_e
        if cur_type is not None:
            s, e = _clean_span(text, cur_s, cur_e)
            if e > s:
                out.append(Concept(text=text[s:e], type=cur_type, position=(s, e)))
        cur_type = cur_s = cur_e = None

    for (cs, ce), lid in zip(offsets, label_ids):
        if cs == ce:  # special token
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab == "O":
            flush()
        elif lab.startswith("B-"):
            flush()
            cur_type = lab[2:]
            cur_s, cur_e = cs, ce
        else:  # I-
            t = lab[2:]
            if cur_type == t:
                cur_e = ce
            else:  # I- mồ côi -> coi như bắt đầu mới (robust)
                flush()
                cur_type, cur_s, cur_e = t, cs, ce
    flush()
    return out


# ------------------------- Tagger inference -------------------------
class EncoderTagger:
    """Bọc model token-classification đã train. Implements ner.base.ConceptTagger.

    Sliding window để xử lý văn bản dài (>max_length token).
    """

    def __init__(self, model, tokenizer, device=None, max_length=256, stride=64):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.stride = stride

    @classmethod
    def from_pretrained(cls, path, **kw):
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForTokenClassification.from_pretrained(path)
        return cls(mdl, tok, **kw)

    def tag(self, text: str) -> list[Concept]:
        import torch

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            max_length=self.max_length,
            stride=self.stride,
            truncation=True,
            padding=False,
        )
        # gom nhãn theo offset token (window sớm hơn thắng khi trùng)
        by_offset: dict[tuple[int, int], int] = {}
        order: list[tuple[int, int]] = []
        for w in range(len(enc["input_ids"])):
            ids = torch.tensor([enc["input_ids"][w]], device=self.device)
            mask = torch.tensor([enc["attention_mask"][w]], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits[0]
            preds = logits.argmax(-1).tolist()
            for (cs, ce), lid in zip(enc["offset_mapping"][w], preds):
                if cs == ce:
                    continue
                key = (cs, ce)
                if key not in by_offset:
                    by_offset[key] = lid
                    order.append(key)
        order.sort()
        offsets = order
        label_ids = [by_offset[k] for k in order]
        return decode_spans(text, offsets, label_ids)
