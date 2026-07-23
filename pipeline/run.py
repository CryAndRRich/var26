"""Orchestrator: input_dir -> output_dir (100 file .json). Ghép các module.

Ở BƯỚC 3 mới nối khung; NER/mapping còn stub. Chạy được end-to-end khi
truyền tagger/mapper cụ thể; nếu không, sinh output rỗng hợp lệ để test I/O & metric.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..assertion.rules import annotate as annotate_assertions
from ..io.dataset import load_inputs
from ..io.schema import (
    Concept,
    TYPES_WITH_CANDIDATES,
    save_concepts,
    validate_concept,
)
from ..mapping.base import CandidateMapper
from ..ner.base import ConceptTagger


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    tagger: Optional[ConceptTagger] = None,
    mapper: Optional[CandidateMapper] = None,
    do_assertions: bool = True,
    asserter=None,
) -> None:
    """Ghép 3 module -> output/.
    - tagger: ConceptTagger (EncoderTagger). None -> concept rỗng.
    - asserter: object có .annotate(text, concepts) (vd LLMAsserter). None -> rule mặc định.
    - mapper: CandidateMapper (RetrievalRerankMapper / LookupMapper).
    """
    annotate = asserter.annotate if asserter is not None else annotate_assertions
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(input_dir)

    for i, text in inputs.items():
        concepts: list[Concept] = tagger.tag(text) if tagger else []
        if do_assertions and concepts:
            annotate(text, concepts)
        if mapper:
            for c in concepts:
                if c.type in TYPES_WITH_CANDIDATES:
                    c.candidates = mapper.map_concept(text, c)
        # kiểm tra tính hợp lệ (offset khớp) trước khi ghi
        for c in concepts:
            errs = validate_concept(c, text)
            if errs:
                raise ValueError(f"[{i}] concept không hợp lệ: {errs}")
        save_concepts(concepts, out / f"{i}.json")


def package_submission(output_dir: str | Path, zip_path: str | Path) -> None:
    """Nén output_dir thành output.zip đúng cấu trúc nộp bài."""
    import zipfile

    out = Path(output_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for jf in sorted(out.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            zf.write(jf, arcname=f"output/{jf.name}")
