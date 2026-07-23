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


# ------------------------- Sơ đồ nhãn BIO (tổng quát) -------------------------
class LabelScheme:
    """Sơ đồ BIO cho MỘT tập type bất kỳ. Dùng cho:
      - EXP1/Stage B: tập 5 type cuộc thi (mặc định, xem COMPETITION bên dưới).
      - EXP2/Stage A: tập type gốc của data ngoài (PhoNER/ViMQ) — khác hoàn toàn.
    Tách scheme ra để pretrain 2 giai đoạn KHÔNG dạy nhầm nhãn (2 head khác nhau).
    """

    def __init__(self, types):
        self.types = list(types)
        labels = ["O"]
        for t in self.types:
            labels.append(f"B-{t}")
            labels.append(f"I-{t}")
        self.labels = labels
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.o_id = self.label2id["O"]

    def __len__(self) -> int:
        return len(self.labels)

    def align(self, offsets, spans, ignore_index: int = -100) -> list[int]:
        """offsets: (char_start, char_end) mỗi token (special token = (0,0)).
        spans: (start, end, type). Trả nhãn id BIO; special token = ignore_index.
        Span có type NGOÀI scheme này bị bỏ qua (an toàn khi trộn nguồn)."""
        labels = [ignore_index if cs == ce else self.o_id for (cs, ce) in offsets]
        for (s, e, t) in sorted(spans, key=lambda x: x[0]):
            if f"B-{t}" not in self.label2id:
                continue
            b_id, i_id = self.label2id[f"B-{t}"], self.label2id[f"I-{t}"]
            first = True
            for idx, (cs, ce) in enumerate(offsets):
                if cs == ce:
                    continue
                if cs < e and ce > s:  # token giao với span
                    labels[idx] = b_id if first else i_id
                    first = False
        return labels

    def decode(self, text: str, offsets, label_ids) -> list[Concept]:
        """Ghép run B/I liên tiếp cùng type -> Concept (theo id2label của scheme này)."""
        return _decode_with(self.id2label, text, offsets, label_ids)


# Scheme mặc định = 5 type cuộc thi. Giữ các tên cũ (LABELS, align_labels...) để
# notebook EXP1 và code hiện tại không phải sửa.
COMPETITION = LabelScheme(TYPES)


def build_labels() -> list[str]:
    return list(COMPETITION.labels)


LABELS = COMPETITION.labels
LABEL2ID = COMPETITION.label2id
ID2LABEL = COMPETITION.id2label
O_ID = COMPETITION.o_id


def align_labels(offsets, spans, ignore_index: int = -100) -> list[int]:
    """Alias tương thích ngược -> COMPETITION.align (5 type cuộc thi)."""
    return COMPETITION.align(offsets, spans, ignore_index=ignore_index)


# ------------------------- featurize (dùng chung notebook + external) -------------------------
def build_features(examples, tok, scheme: "LabelScheme | None" = None,
                   max_length: int = 256, stride: int = 64) -> list[dict]:
    """examples: iterable (text, spans) với spans=[(start,end,type),...].
    Trả list feature {input_ids, attention_mask, labels} (sliding-window).
    CHỈ dùng tokenizer (CPU-safe) — vòng train đặt ở notebook."""
    scheme = scheme or COMPETITION
    feats: list[dict] = []
    for text, spans in examples:
        enc = tok(text, return_offsets_mapping=True, return_overflowing_tokens=True,
                  max_length=max_length, stride=stride, truncation=True, padding=False)
        for w in range(len(enc["input_ids"])):
            feats.append({
                "input_ids": enc["input_ids"][w],
                "attention_mask": enc["attention_mask"][w],
                "labels": scheme.align(enc["offset_mapping"][w], spans),
            })
    return feats


# ------------------------- token labels -> char spans -------------------------
def _clean_span(text: str, s: int, e: int) -> tuple[int, int]:
    """Bỏ khoảng trắng đầu/cuối để text[s:e] gọn (khớp cách gán nhãn của đề)."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _decode_with(id2label: dict, text: str, offsets, label_ids) -> list[Concept]:
    """Lõi decode BIO -> Concept theo bảng id2label truyền vào."""
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
        lab = id2label.get(int(lid), "O")
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


def decode_spans(text: str, offsets, label_ids) -> list[Concept]:
    """Decode theo scheme cuộc thi (tương thích ngược)."""
    return _decode_with(ID2LABEL, text, offsets, label_ids)


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
        # decode theo id2label của chính model (khớp scheme đã train, kể cả model
        # Stage A có tập type khác). Ép key int cho chắc.
        cfg = getattr(model, "config", None)
        self.id2label = {int(k): v for k, v in cfg.id2label.items()} if cfg and getattr(cfg, "id2label", None) else ID2LABEL

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
        return _decode_with(self.id2label, text, offsets, label_ids)
