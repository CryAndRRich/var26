"""Interface cho NER + phân loại type. Cài đặt cụ thể (encoder / LLM) đặt sau.

Ràng buộc: bản chạy nặng (transformer) chỉ chạy trên notebooks/ (Kaggle GPU).
Module này chỉ định nghĩa hợp đồng (contract) để pipeline gọi thống nhất.
"""
from __future__ import annotations

from typing import Protocol

from ..io.schema import Concept


class ConceptTagger(Protocol):
    """Nhận text thô -> list Concept (text, type, position).

    KHÔNG điền candidates/assertions ở đây; đó là việc của module mapping/assertion.
    Yêu cầu: position phải khớp input[start:end] == text (offset ký tự gốc).
    """

    def tag(self, text: str) -> list[Concept]: ...


# --- Kế hoạch cài đặt (BƯỚC 4+) ---
# 1. EncoderTagger: XLM-R / PhoBERT token-classification (BIO + type). Ưu offset chuẩn.
#    - train trên data/train + train_labels; đo bằng eval.metrics.
# 2. LLMTagger: Qwen2.5-7B sinh JSON span, kèm bước align span -> offset gốc.
# 3. HybridTagger: encoder làm nền, LLM vá các loại khó (KẾT_QUẢ_XÉT_NGHIỆM, span dài).
