"""Assertion bằng LLM (Qwen2.5-7B QLoRA) — EXP3.

Phần TÁI SỬ DỤNG & CPU-safe ở đây: trích ngữ cảnh, dựng prompt, parse output,
sinh cặp (prompt, target) để train. Nạp/generate model là lazy (chỉ import
torch/transformers khi thực sự chạy) — vòng QLoRA đặt ở notebooks/ (Kaggle GPU).

Bài toán: cho 1 concept (CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG) + ngữ cảnh câu chứa nó,
chọn tập con trong {isNegated, isFamily, isHistorical} áp dụng (có thể rỗng).
Precision-first: đa số concept KHÔNG có assertion (rỗng) — model phải học điều đó.

So với rule (`rules.py`): rule là baseline; LLM kỳ vọng bắt ngữ cảnh phủ định/
tiền sử/gia đình khó (mệnh đề xa, cách diễn đạt lạ) mà rule cứng bỏ sót.
"""
from __future__ import annotations

import json
import re

from ..io.schema import ASSERTIONS, Concept, TYPES_WITH_ASSERTIONS

# Đánh dấu mention trong ngữ cảnh để model biết đang hỏi về cụm nào.
MARK_L, MARK_R = "«", "»"

_DEF = (
    "isNegated: khái niệm bị PHỦ ĐỊNH/không có/loại trừ/âm tính.\n"
    "isHistorical: thuộc TIỀN SỬ/bệnh cũ/đã xảy ra trước đợt khám này.\n"
    "isFamily: KHÔNG phải của bệnh nhân mà của NGƯỜI NHÀ (bố/mẹ/anh/chị...)."
)

SYSTEM_PROMPT = (
    "Bạn là trợ lý y khoa. Với khái niệm y tế được đánh dấu trong ngữ cảnh, hãy xác "
    "định các nhãn ngữ cảnh (assertion) áp dụng cho nó. Chỉ chọn trong ba nhãn sau, "
    "và CHỈ khi ngữ cảnh nêu rõ; nếu không có nhãn nào phù hợp, trả về danh sách rỗng.\n"
    f"{_DEF}\n"
    'Trả lời DUY NHẤT một mảng JSON, ví dụ: ["isNegated"] hoặc [].'
)


def context_window(text: str, concept: Concept, before: int = 160, after: int = 60) -> str:
    """Ngữ cảnh quanh mention: giới hạn trong DÒNG chứa nó (không rò sang mục khác),
    cắt cửa sổ ký tự để prompt gọn, và đánh dấu mention bằng «»."""
    s, e = concept.position
    line_start = text.rfind("\n", 0, s) + 1
    line_end = text.find("\n", e)
    if line_end == -1:
        line_end = len(text)
    ws = max(line_start, s - before)
    we = min(line_end, e + after)
    left, mid, right = text[ws:s], text[s:e], text[e:we]
    return f"{left}{MARK_L}{mid}{MARK_R}{right}".strip()


def build_prompt(text: str, concept: Concept) -> str:
    """Prompt người-dùng (chưa gồm system). Notebook ghép theo chat template của model."""
    ctx = context_window(text, concept)
    return (
        f"Loại khái niệm: {concept.type}\n"
        f"Khái niệm: {concept.text}\n"
        f"Ngữ cảnh: {ctx}\n"
        "Các nhãn áp dụng (mảng JSON):"
    )


def parse_output(s: str) -> list[str]:
    """Parse output model -> list assertion hợp lệ, khử trùng lặp, giữ thứ tự chuẩn.
    Bền vững với text thừa quanh JSON; fallback dò tên nhãn trực tiếp."""
    found: set[str] = set()
    m = re.search(r"\[.*?\]", s, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                found = {x for x in arr if isinstance(x, str) and x in ASSERTIONS}
        except json.JSONDecodeError:
            pass
    if not found:  # fallback: tên nhãn xuất hiện trực tiếp
        found = {a for a in ASSERTIONS if a in s}
    return [a for a in ASSERTIONS if a in found]  # thứ tự ổn định theo ASSERTIONS


def target_json(concept: Concept) -> str:
    """Chuỗi target để train: JSON các assertion gold theo thứ tự chuẩn."""
    gold = [a for a in ASSERTIONS if a in set(concept.assertions)]
    return json.dumps(gold, ensure_ascii=False)


def build_examples(labeled: list[tuple[str, list[Concept]]]) -> list[dict]:
    """Từ train (text, concepts) -> [{system, prompt, target, type}] cho QLoRA.
    Chỉ lấy concept thuộc TYPES_WITH_ASSERTIONS."""
    out: list[dict] = []
    for text, concepts in labeled:
        for c in concepts:
            if c.type not in TYPES_WITH_ASSERTIONS:
                continue
            out.append({
                "system": SYSTEM_PROMPT,
                "prompt": build_prompt(text, c),
                "target": target_json(c),
                "type": c.type,
            })
    return out


class LLMAsserter:
    """Bọc model chat đã (QLoRA) finetune. Interface như rules.annotate.

    Dùng ở inference (notebooks/). Local CPU chỉ nên gọi build_prompt/parse_output.
    """

    def __init__(self, model, tokenizer, device=None, max_new_tokens: int = 16,
                 system_prompt: str = SYSTEM_PROMPT):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def _generate(self, user_prompt: str) -> str:
        from ..llm_util import chat_generate

        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}]
        return chat_generate(self.model, self.tokenizer, msgs, self.max_new_tokens)

    def infer_assertions(self, text: str, concept: Concept) -> list[str]:
        if concept.type not in TYPES_WITH_ASSERTIONS:
            return []
        return parse_output(self._generate(build_prompt(text, concept)))

    def annotate(self, text: str, concepts: list[Concept]) -> list[Concept]:
        for c in concepts:
            c.assertions = self.infer_assertions(text, c)
        return concepts
