# var26 — Medical Concept Detection & Normalization (VAR2026)

Code chính cho bài toán phát hiện & chuẩn hóa khái niệm y khoa tiếng Việt.
Cho mỗi input `.txt` sinh `.json` gồm các khái niệm: `text`, `type` (5 loại),
`position` (offset ký tự), `assertions`, `candidates` (ICD-10 / RxNorm).
Tối ưu `final = 0.3·text + 0.3·assertions + 0.4·candidates`.

## Cài đặt
```bash
pip install -r requirements.txt   # rank-bm25, transformers... (tùy phần dùng)
```

## Cấu trúc package (repo root = package `var26`)
| Module | Vai trò |
|---|---|
| `io/` | `schema.py` (Concept, đọc/ghi JSON, validate offset), `dataset.py` |
| `ner/` | `base.py` (interface `ConceptTagger`), `gazetteer.py` (baseline CPU) |
| `assertion/` | `rules.py` (NegEx/ConText Việt hóa, precision-first) |
| `mapping/` | `icd_index.py` (ICD-10), `lookup.py` (baseline có GATE), `base.py` (interface) |
| `eval/` | `metrics.py` (proxy scorer: text WER + Jaccard assert/cand) |
| `pipeline/` | `run.py` (`input_dir -> output/` + `package_submission`) |

## Dùng trên Kaggle (GPU)
```python
!git clone https://github.com/CryAndRRich/var26.git
import sys; sys.path.insert(0, ".")   # thư mục CHỨA var26/ (không phải var26/ chính nó)
from var26.pipeline.run import run, package_submission
from var26.ner.gazetteer import GazetteerTagger   # hoặc encoder tagger mới
```
Lưu ý: repo clone ra thư mục `var26/` chính là package → thêm THƯ MỤC CHA vào `sys.path` rồi `import var26`.
Dữ liệu (`data/`) upload dưới dạng Kaggle Dataset và mount, KHÔNG hard-code path.

## Ghi chú thiết kế
- Không hard-code path tuyệt đối; mọi path do caller truyền vào.
- `position` tính trên text RAW (không normalize) để offset khớp `input[start:end]==text`.
- NER encoder / LLM rerank (nặng) chạy trên Kaggle; phần rule/exact/metric chạy CPU.
