"""Nạp data NER ngoài đã convert (PhoNER, ViMQ) cho pretrain 2 giai đoạn (EXP2).

Nguồn: `data_external/processed/{phoner,vimq}.jsonl` — sinh bởi
`scripts/convert_external_ner.py`. Mỗi dòng:
    {"text": str, "spans": [[start,end,type],...], "source": str, "split": str}

QUAN TRỌNG (vì sao 2 giai đoạn, không merge nhãn):
  Nhãn gốc 2 bộ này KHÔNG khớp 5 type cuộc thi (SYMPTOM_AND_DISEASE gộp
  TRIỆU_CHỨNG+CHẨN_ĐOÁN; không có KẾT_QUẢ_XÉT_NGHIỆM...). Nếu ép về scheme cuộc
  thi sẽ dạy nhầm. Nên Stage A train với `LabelScheme(external_types(...))` (tập
  type GỐC), chỉ để encoder học biểu diễn domain + nhận biên thực thể; Stage B mới
  thay head, finetune 5-type trên train thi thật.

Path do caller truyền (không hard-code) — hợp rule dự án.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_external_jsonl(*paths: str | Path, splits: set[str] | None = None) -> list[dict]:
    """Đọc 1+ file jsonl -> list record. `splits` lọc theo split (vd {'train','dev'})."""
    records: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if splits is not None and obj.get("split") not in splits:
                    continue
                records.append(obj)
    return records


def external_types(records: list[dict]) -> list[str]:
    """Tập type xuất hiện trong records (sắp xếp ổn định để id nhãn tái lập được)."""
    types = {t for r in records for _s, _e, t in r["spans"]}
    return sorted(types)


def to_examples(records: list[dict]) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """-> [(text, [(start,end,type),...]), ...] cho `encoder.build_features`."""
    return [(r["text"], [(s, e, t) for s, e, t in r["spans"]]) for r in records]


def default_paths(processed_dir: str | Path) -> list[str]:
    """Trả các file jsonl tồn tại trong thư mục processed (phoner, vimq)."""
    d = Path(processed_dir)
    return [str(d / f"{name}.jsonl") for name in ("phoner", "vimq") if (d / f"{name}.jsonl").exists()]
