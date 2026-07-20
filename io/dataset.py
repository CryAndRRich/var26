"""Nạp dataset train/test. Path do caller truyền (không hard-code)."""
from __future__ import annotations

from pathlib import Path

from .schema import Concept, load_concepts, read_input_text


def list_ids(input_dir: str | Path) -> list[str]:
    """Trả id sắp theo số (1,2,...,100) từ các file {id}.txt."""
    ids = [p.stem for p in Path(input_dir).glob("*.txt")]
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def load_inputs(input_dir: str | Path) -> dict[str, str]:
    """{id: text} — giữ nguyên ký tự gốc."""
    d = Path(input_dir)
    return {i: read_input_text(d / f"{i}.txt") for i in list_ids(d)}


def load_labeled(
    input_dir: str | Path, label_dir: str | Path
) -> dict[str, tuple[str, list[Concept]]]:
    """{id: (text, [Concept])} cho tập có nhãn (train)."""
    d, ld = Path(input_dir), Path(label_dir)
    out: dict[str, tuple[str, list[Concept]]] = {}
    for i in list_ids(d):
        out[i] = (read_input_text(d / f"{i}.txt"), load_concepts(ld / f"{i}.json"))
    return out
