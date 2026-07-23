"""LLM rerank + GATE cho code-mapping (EXP4) — tầng cuối kiến trúc 3 tầng.

Luồng 1 concept CHẨN_ĐOÁN/THUỐC:
  exact (ICD/RxNorm) --unique?--> ứng viên mạnh
  + retrieval top-k (BM25/dense)         } gộp danh sách ứng viên
  -> LLM rerank chọn 1 mã HOẶC "NONE"  (NONE = GATE: để candidates=[])

GATE là quyết định điểm lớn nhất (>80% CHẨN_ĐOÁN & ~70% THUỐC để rỗng). Hai lớp gate:
  (a) prior học từ train: (type, mention) hầu như không có mã -> bỏ qua sớm (khỏi gọi LLM).
  (b) tuỳ chọn "NONE" của LLM rerank.

Phần dựng/parse prompt CPU-safe; nạp/generate LLM là lazy (chạy ở notebooks/).
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..io.schema import Concept, TYPES_WITH_CANDIDATES
from .icd_index import normalize

NONE_TOKEN = "NONE"

SYSTEM_PROMPT = (
    "Bạn là chuyên gia mã hoá y khoa. Cho một khái niệm y tế và danh sách mã ứng viên "
    "(kèm mô tả), hãy chọn ĐÚNG MỘT mã khớp nhất về mặt lâm sàng. Nếu KHÔNG mã nào thực "
    f"sự khớp, trả về {NONE_TOKEN}. Chỉ in ra mã đã chọn hoặc {NONE_TOKEN}, không giải thích."
)


def build_prompt(concept: Concept, candidates: list[tuple[str, str]], context: str = "") -> str:
    """candidates: [(code, description), ...] (đã lấy top-k)."""
    lines = [f"{code}\t{desc}" for code, desc in candidates]
    ctx = f"Ngữ cảnh: {context}\n" if context else ""
    return (
        f"Loại: {concept.type}\n"
        f"Khái niệm: {concept.text}\n"
        f"{ctx}"
        "Mã ứng viên:\n" + "\n".join(lines) + "\n"
        f"Mã khớp nhất (hoặc {NONE_TOKEN}):"
    )


def parse_output(s: str, valid_codes: set[str]) -> str | None:
    """Trích mã hợp lệ từ output; NONE/không khớp -> None (GATE để rỗng)."""
    s = s.strip()
    if NONE_TOKEN.lower() in s.lower() and not any(c in s for c in valid_codes):
        return None
    for code in sorted(valid_codes, key=len, reverse=True):  # khớp mã dài trước
        if re.search(r"(?<![\w.])" + re.escape(code) + r"(?![\w.])", s):
            return code
    return None


def build_gate(labeled: list[tuple[str, list[Concept]]], min_ratio: float = 0.15) -> dict:
    """Học prior GATE từ train: với mỗi (type, mention_chuẩn_hoá) tỉ lệ CÓ mã;
    và base-rate theo type. Trả dict để `RetrievalRerankMapper` lọc sớm.
    min_ratio thấp (0.15) vì đây là prior "đáng xét", quyết định cuối để LLM chọn NONE."""
    seen: dict[tuple[str, str], int] = defaultdict(int)
    coded: dict[tuple[str, str], int] = defaultdict(int)
    seen_t: dict[str, int] = defaultdict(int)
    coded_t: dict[str, int] = defaultdict(int)
    for _text, concepts in labeled:
        for c in concepts:
            if c.type not in TYPES_WITH_CANDIDATES:
                continue
            k = (c.type, normalize(c.text))
            seen[k] += 1
            seen_t[c.type] += 1
            if c.candidates:
                coded[k] += 1
                coded_t[c.type] += 1
    ratio = {k: coded[k] / n for k, n in seen.items()}
    base = {t: coded_t[t] / n for t, n in seen_t.items()}
    return {"ratio": ratio, "base": base, "min_ratio": min_ratio}


class LLMReranker:
    """Bọc LLM chat (Qwen2.5-7B) cho rerank. Lazy — chạy ở notebooks/."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 16, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def _generate(self, user_prompt: str) -> str:
        import torch

        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    def rerank(self, concept: Concept, candidates: list[tuple[str, str]], context: str = "") -> str | None:
        if not candidates:
            return None
        out = self._generate(build_prompt(concept, candidates, context))
        return parse_output(out, {c for c, _ in candidates})


class RetrievalRerankMapper:
    """CandidateMapper: exact + retrieval -> LLM rerank + GATE. Implements mapping.base.

    index_by_type / retriever_by_type: {type: obj}. reranker: LLMReranker|None
    (None -> chỉ nhận exact-unique, làm baseline không-LLM). gate: build_gate(...)|None.
    """

    def __init__(self, index_by_type: dict, retriever_by_type: dict,
                 reranker: "LLMReranker | None" = None, gate: dict | None = None,
                 topk: int = 10, use_context: bool = True):
        self.index_by_type = index_by_type
        self.retriever_by_type = retriever_by_type
        self.reranker = reranker
        self.gate = gate
        self.topk = topk
        self.use_context = use_context

    def _passes_prior(self, concept: Concept) -> bool:
        if not self.gate:
            return True
        k = (concept.type, normalize(concept.text))
        r = self.gate["ratio"].get(k)
        if r is not None:
            return r >= self.gate["min_ratio"]
        return self.gate["base"].get(concept.type, 1.0) >= self.gate["min_ratio"]

    def map_concept(self, text: str, concept: Concept) -> list[str]:
        if concept.type not in TYPES_WITH_CANDIDATES:
            return []
        if not self._passes_prior(concept):
            return []  # GATE prior: loại nhóm gần như luôn rỗng

        index = self.index_by_type.get(concept.type)
        retriever = self.retriever_by_type.get(concept.type)

        # ứng viên: exact (nếu có) + retrieval top-k
        cand: list[tuple[str, str]] = []
        seen: set[str] = set()
        if index is not None:
            for code in index.exact(concept.text):
                if code not in seen:
                    cand.append((code, concept.text)); seen.add(code)
        if retriever is not None:
            for code, _score, desc in retriever.search(concept.text, k=self.topk):
                if code not in seen:
                    cand.append((code, desc)); seen.add(code)

        if not cand:
            return []
        if self.reranker is None:
            # baseline không-LLM: chỉ nhận khi exact duy nhất
            ex = index.exact(concept.text) if index is not None else []
            return ex[:1] if len(ex) == 1 else []

        ctx = ""
        if self.use_context:
            s, e = concept.position
            ls = text.rfind("\n", 0, s) + 1
            le = text.find("\n", e)
            ctx = text[ls:(le if le != -1 else len(text))].strip()[:200]
        code = self.reranker.rerank(concept, cand[: self.topk], context=ctx)
        return [code] if code else []
