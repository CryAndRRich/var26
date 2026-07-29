"""GATE bằng encoder + mapper xâu chuỗi đúng thứ tự — EXP4b.

## Vì sao có module này (bằng chứng từ EXP4, xem docs/results/2026-07-23_EXP4_candidates.md)
EXP4 áp retrieval/rerank cho MỌI concept → 0.329 tụt xuống 0.182. Chẩn đoán:
  - Retriever KHÔNG yếu: BM25 recall@10 = 81% (ICD) / 69% (RxNorm).
  - Nút thắt là THIẾU GATE: chỉ ~18% CHẨN_ĐOÁN và ~24% THUỐC thực sự có mã, nên gán mã
    cho phần còn lại làm phình union → Jaccard/file sụp.
  - Trần đo được (gate hoàn hảo): lookup 0.422 · +BM25 fallback 0.459 · +rerank hoàn hảo 0.620.
⟹ Thứ tự BẮT BUỘC: **GATE trước**, rồi mới chọn mã.

Gate học bằng logistic + feature thủ công KHÔNG thắng lookup (0.334 vs 0.349) vì chỉ học lại
`min_code_ratio`. Nhưng ngữ cảnh mục CÓ tín hiệu (`tiền sử` -1.60, `header chẩn đoán` +0.54)
⟹ dùng ENCODER đọc ngữ cảnh, đúng công thức đã thắng đậm ở EXP3b (assertion concept 0.055→0.414).

Phần ở đây CPU-safe; vòng train đặt ở notebooks/ (Kaggle GPU).
"""
from __future__ import annotations

from ..io.schema import Concept, TYPES_WITH_CANDIDATES
# Dùng CHUNG bộ dựng ngữ cảnh với assertion-encoder (type | section path | «mention»)
# để hai module so được với nhau và cùng thấy heading — xem var26/text/context.py.
from ..text.context import build_input as _ctx_input
from ..text.context import build_inputs as _ctx_inputs


def build_input(text: str, concept: Concept, **kw) -> str:
    """Input cho gate: `type | section path | ngữ cảnh có «mention»` (giống EXP3b)."""
    return _ctx_input(text, concept, **kw)


def build_inputs(text: str, concepts: list[Concept], **kw) -> list[str]:
    """Như `build_input` cho nhiều concept cùng văn bản (phân tích heading 1 lần)."""
    return _ctx_inputs(text, concepts, **kw)


def build_examples(labeled: list[tuple[str, list[Concept]]], **kw) -> list[dict]:
    """(text, concepts) -> [{'text', 'label'(0/1), 'type'}]; label=1 nếu gold CÓ mã."""
    out: list[dict] = []
    for text, concepts in labeled:
        todo = [c for c in concepts if c.type in TYPES_WITH_CANDIDATES]
        for c, inp in zip(todo, build_inputs(text, todo, **kw)):
            out.append({"text": inp, "label": 1 if c.candidates else 0, "type": c.type})
    return out


def featurize(examples: list[dict], tok, max_length: int = 224) -> list[dict]:
    """Tokenize -> feature cho HF Trainer (num_labels=2, CrossEntropy, label int)."""
    feats: list[dict] = []
    for e in examples:
        enc = tok(e["text"], truncation=True, max_length=max_length)
        feats.append({"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"],
                      "labels": int(e["label"])})
    return feats


class EncoderGate:
    """P(concept này ĐƯỢC gán mã). Model 2 lớp; trả softmax của lớp 1."""

    def __init__(self, model, tokenizer, threshold: float = 0.5,
                 batch_size: int = 64, max_length: int = 224,
                 ctx_kw: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.batch_size = batch_size
        # 160 -> 224: input giờ có thêm section path + vài dòng ngữ cảnh.
        self.max_length = max_length
        # PHẢI khớp cấu hình dùng lúc train.
        self.ctx_kw = dict(ctx_kw or {})
        self.model.eval()

    def predict_proba(self, texts: list[str]) -> list[float]:
        import torch

        if not texts:
            return []
        device = getattr(self.model, "device", None) or "cpu"
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], return_tensors="pt",
                                 padding=True, truncation=True, max_length=self.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.extend(torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist())
        return out

    def proba_for(self, text: str, concepts: list[Concept]) -> dict[int, float]:
        """{id(concept): prob} cho các concept thuộc loại có candidates."""
        todo = [c for c in concepts if c.type in TYPES_WITH_CANDIDATES]
        probs = self.predict_proba(build_inputs(text, todo, **self.ctx_kw))
        return {id(c): p for c, p in zip(todo, probs)}

    def passes(self, text: str, concept: Concept) -> bool:
        if concept.type not in TYPES_WITH_CANDIDATES:
            return False
        return self.predict_proba([build_input(text, concept, **self.ctx_kw)])[0] >= self.threshold


class GatedMapper:
    """GATE -> chọn mã theo thứ tự ưu tiên precision. Implements mapping.base.CandidateMapper.

    Chuỗi chọn mã (dừng ở tầng đầu tiên có kết quả):
      1. `lookup` (memorize từ train) — tín hiệu MẠNH NHẤT theo EXP4 (0.329 một mình).
         RetrievalRerankMapper cũ KHÔNG dùng tầng này — đó là một lý do nó kém.
      2. exact match trên KB (ICD/RxNorm).
      3. retrieval top-k -> `reranker` chọn 1 (hoặc top-1 nếu không có reranker).

    `use_retrieval=False` để đo riêng đóng góp của gate+lookup.
    """

    def __init__(self, gate: "EncoderGate | None", lookup=None, index_by_type: dict | None = None,
                 retriever_by_type: dict | None = None, reranker=None,
                 topk: int = 10, use_retrieval: bool = True):
        self.gate = gate
        self.lookup = lookup
        self.index_by_type = index_by_type or {}
        self.retriever_by_type = retriever_by_type or {}
        self.reranker = reranker
        self.topk = topk
        self.use_retrieval = use_retrieval

    # ---- chọn mã (KHÔNG xét gate) — tách riêng để tune threshold không phải chạy lại ----
    def select_codes(self, text: str, concept: Concept) -> list[str]:
        if concept.type not in TYPES_WITH_CANDIDATES:
            return []
        if self.lookup is not None:
            got = self.lookup.map_concept(text, concept)
            if got:
                return got
        index = self.index_by_type.get(concept.type)
        if index is not None:
            ex = index.exact(concept.text)
            if len(ex) == 1:          # chỉ nhận khi DUY NHẤT -> tránh nhiễu
                return ex
        if not self.use_retrieval:
            return []
        retr = self.retriever_by_type.get(concept.type)
        if retr is None:
            return []
        hits = retr.search(concept.text, k=self.topk)
        if not hits:
            return []
        if self.reranker is None:
            return [hits[0][0]]
        code = self.reranker.rerank(concept, [(c, d) for c, _s, d in hits],
                                    context=_line_of(text, concept))
        return [code] if code else []

    def map_concept(self, text: str, concept: Concept) -> list[str]:
        if concept.type not in TYPES_WITH_CANDIDATES:
            return []
        if self.gate is not None and not self.gate.passes(text, concept):
            return []
        return self.select_codes(text, concept)


def line_of(text: str, concept: Concept, limit: int = 200) -> str:
    """Dòng chứa mention (cắt `limit` ký tự) — dùng làm ngữ cảnh cho LLM rerank."""
    s, e = concept.position
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return text[ls:(le if le != -1 else len(text))].strip()[:limit]


_line_of = line_of   # tương thích ngược


def tune_gate_threshold(dev_labeled, gate: "EncoderGate", mapper: "GatedMapper",
                        key_mode: str = "concept_first", grid=None, verbose: bool = True):
    """Chọn ngưỡng gate tối đa CHÍNH `candidates_score` trên dev.

    Hiệu quả: prob của gate và mã chọn được TÍNH MỘT LẦN cho mỗi concept, sau đó chỉ thay
    ngưỡng ⟹ quét cả grid không cần chạy lại model/retrieval (kể cả khi có LLM rerank).
    Lưu ý dùng `assertions`-style: gọi `candidates_score_sample` trực tiếp để bỏ text-score (WER).

    `key_mode="concept_first"` (mặc định) = 2/3 "concept" + 1/3 "value". Đề (mục 5a) mô tả
    sai `type` bị "tính 2 lần, mỗi lần 0 điểm" — chỉ đúng nếu scorer gióng theo từng khái
    niệm ⟹ nghiêng "concept". Tối ưu riêng một mode là đánh cược (bài học EXP5).

    Grid CÓ 0.0 — nghĩa là "không gate". Ở EXP4b grid bắt đầu từ 0.05 nên với biến thể
    `gate + lookup` tuner chọn đúng biên dưới, dấu hiệu tối ưu thật nằm ngoài grid.
    Trả (threshold, best_score).
    """
    import copy

    from ..eval.metrics import candidates_score_sample

    grid = grid if grid is not None else [0.0] + [i / 20 for i in range(1, 20)]
    from ..eval.metrics import CONCEPT_FIRST_W
    if key_mode == "concept_first":
        weights = dict(CONCEPT_FIRST_W)
    elif key_mode == "balanced":
        weights = {"value": 0.5, "concept": 0.5}
    else:
        weights = {key_mode: 1.0}
    modes = tuple(weights)

    cached = []   # (gold, {id: (prob, codes)})
    for text, gold in dev_labeled:
        probs = gate.proba_for(text, gold)
        info = {}
        for c in gold:
            if c.type in TYPES_WITH_CANDIDATES:
                info[id(c)] = (probs.get(id(c), 0.0), mapper.select_codes(text, c))
        cached.append((gold, info))

    def evaluate(th: float) -> float:
        total = 0.0
        for m in modes:
            num = den = 0.0
            for gold, info in cached:
                pred = []
                for c in gold:
                    c2 = copy.copy(c)
                    if c.type in TYPES_WITH_CANDIDATES:
                        p, codes = info[id(c)]
                        c2.candidates = codes if p >= th else []
                    else:
                        c2.candidates = []
                    pred.append(c2)
                j, w = candidates_score_sample(gold, pred, m)
                num += j * w
                den += w
            total += weights[m] * (num / den if den else 0.0)
        return total

    best_t, best_s = grid[0], -1.0
    for t in grid:
        s = evaluate(t)
        if s > best_s:
            best_t, best_s = t, s
    if verbose:
        print(f"  gate threshold tốt nhất = {best_t:.2f} -> {key_mode} cand = {best_s:.4f}")
    return best_t, best_s
