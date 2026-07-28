"""Nạp & tra cứu RxNorm từ DB local (build offline từ RRF) — cho THUỐC.

DB dạng CSV cột: rxcui,name (build bởi `scripts/build_rxnorm_db.py` từ RXNCONSO.RRF).
KHÔNG gọi RxNav API lúc inference (đề cấm API ngoài); API chỉ dùng offline khi build.

Song song với ICDIndex: cùng interface (.exact, .entries, len) để retrieval tái dùng.
Thuốc trong train là tên tiếng Anh (prednisone->8640) nên khớp theo tên Anh.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .icd_index import normalize

# Viết tắt lâm sàng hay gặp trong bệnh án -> tên chuẩn. ĐO THỰC TẾ: BM25 trượt các mention
# như "asa 325mg" (ra acetaminophen/ferrous sulfate) vì "asa" không có trong tên RxNorm.
DRUG_ALIASES = {
    "asa": "aspirin",
    "apap": "acetaminophen",
    "hctz": "hydrochlorothiazide",
    "mgso4": "magnesium sulfate",
    "kcl": "potassium chloride",
    "nacl": "sodium chloride",
    "ntg": "nitroglycerin",
    "asa/dipyridamole": "aspirin dipyridamole",
    "abx": "antibiotic",
    "insulin nph": "insulin isophane human",
}


def expand_mention(mention: str) -> str:
    """Thay viết tắt bằng tên chuẩn (giữ nguyên phần liều — mã gold ở mức CÓ LIỀU,
    vd 617317 = 'atorvastatin 20 MG [Lipitor]', nên KHÔNG được bỏ liều)."""
    toks = normalize(mention).split()
    return " ".join(DRUG_ALIASES.get(t, t) for t in toks)


@dataclass
class RxNormEntry:
    code: str          # RXCUI
    description: str    # tên thuốc
    norm: str


class RxNormIndex:
    def __init__(self, entries: list[RxNormEntry]):
        self.entries = entries
        self._by_norm: dict[str, list[str]] = {}
        for e in entries:
            self._by_norm.setdefault(e.norm, []).append(e.code)

    @classmethod
    def from_csv(cls, path: str | Path) -> "RxNormIndex":
        """Đọc DB rxcui,name. Chấp nhận header linh hoạt (rxcui/code, name/str/description)."""
        entries: list[RxNormEntry] = []
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = {f.lower(): f for f in (reader.fieldnames or [])}
            code_col = fields.get("rxcui") or fields.get("code")
            name_col = fields.get("name") or fields.get("str") or fields.get("description")
            for row in reader:
                code = (row.get(code_col) or "").strip()
                name = (row.get(name_col) or "").strip()
                if code and name:
                    entries.append(RxNormEntry(code, name, normalize(name)))
        return cls(entries)

    def exact(self, mention: str) -> list[str]:
        """Khớp chính xác; nếu trượt thì thử lại sau khi mở viết tắt (asa->aspirin...)."""
        hit = self._by_norm.get(normalize(mention), [])
        if hit:
            return hit
        exp = expand_mention(mention)
        return self._by_norm.get(exp, []) if exp != normalize(mention) else []

    def __len__(self) -> int:
        return len(self.entries)
