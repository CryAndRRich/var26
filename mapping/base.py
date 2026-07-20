"""Interface code-mapping 3 tầng có GATE (kiến trúc từ RESEARCH.md §3).

Insight dữ liệu: >80% CHẨN_ĐOÁN và ~70% THUỐC để candidates=[]; gần như luôn <=1 mã.
=> GATE (có nên gán mã) quan trọng hơn cả chất lượng retrieval.
"""
from __future__ import annotations

from typing import Protocol

from ..io.schema import Concept


class CandidateMapper(Protocol):
    """Điền `candidates` cho 1 concept CHẨN_ĐOÁN / THUỐC (in-place)."""

    def map_concept(self, text: str, concept: Concept) -> list[str]: ...


# --- Kiến trúc dự kiến (BƯỚC 4+) ---
# Tầng 0 - GATE: quyết định gán mã hay để []. Học từ train (đặc trưng: type,
#   độ dài mention, có khớp DB không, là mục "tiền sử/chẩn đoán chính"...).
# Tầng 1 - EXACT/near-exact: ICDIndex.exact() cho bệnh; RxNorm exact cho thuốc.
#   -> precision cao, top-1.
# Tầng 2 - RETRIEVAL+RERANK (khi tầng 1 fail): BM25 + SapBERT top-k -> Qwen2.5-7B rerank.
#   -> chạy trên notebooks/ (GPU). Trả top-1 (mở rộng >=2 mã chỉ khi rerank rất tự tin).
#
# RxNorm: cần build DB local từ RxNorm RRF (offline) trước — xem scripts/.
