"""Sinh dữ liệu huấn luyện THEO LAYOUT CỦA TẬP TEST, nhãn chính xác tuyệt đối.

## Vấn đề cần giải

`var26/text/layout.py` ghi số đo: train hầu như không có layout "mục + gạch đầu dòng"
(0.8% ký tự, 0 file có ≥10 bullet) còn test thì 31.7% ký tự / 52 file. Model NER +
assertion + gate đều học trên văn xuôi nên khi gặp danh sách gạch đầu dòng thì lạ hoàn
toàn. Đây là nguyên nhân chính điểm nộp thật (0.284) thấp hơn nhiều DEV nội bộ (0.559).

## Cách làm

Không cần gán nhãn tay: **lấy concept gold của train (đã có `type`, `assertions`,
`candidates`) rồi RENDER lại vào khung mục/bullet giống test**. Vì ta tự đặt chuỗi vào
văn bản nên `position` đúng theo cấu trúc, và nhãn thừa hưởng nguyên từ gold.

Mỗi tiểu mục có "ngữ nghĩa" riêng nên chỉ rút concept KHỚP ngữ nghĩa đó
(vd "Các bệnh lý mạn tính" -> `CHẨN_ĐOÁN` + `isHistorical`; "Triệu chứng khi nhập viện"
-> `TRIỆU_CHỨNG` không assertion). Nhờ vậy model học đúng liên kết *heading -> assertion*
trong layout của test, thay vì phải suy từ cue nằm ngoài cửa sổ ngữ cảnh.

Có chèn **dòng nhiễu** (`Mức độ nghiêm trọng: Không ghi rõ`, `Tần suất: Không ghi rõ`…)
đúng như test — đây là negative quan trọng để model không tag bừa mọi bullet.

Vốn từ heading lấy từ những gì quan sát được ở tập test (đều là từ vựng tóm tắt bệnh án
tiếng Việt thông thường, không phải nội dung test) nên vẫn dùng được cho private test.

CPU-safe. Không đọc file: caller truyền `labeled` vào.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..io.schema import Concept

# ---------------------------------------------------------------------------
# Khung mục: (heading cấp 1, [(heading cấp 2, [(type, assertion-bắt-buộc)])])
# assertion None = concept KHÔNG có assertion nào.
# ---------------------------------------------------------------------------
H_HIST = "isHistorical"
H_NEG = "isNegated"

SECTIONS: list[tuple[list[str], list[tuple[list[str], list[tuple[str, str | None]]]]]] = [
    (["Tiền sử bệnh", "Tiền sử bệnh nội khoa", "Tiền sử bệnh lý", "Tiền sử bệnh nội"], [
        (["Các bệnh lý mạn tính", "Các bệnh lý mãn tính", "Các bệnh mãn tính",
          "Bệnh lý mãn tính", "Tiền sử bệnh nội khoa"],
         [("CHẨN_ĐOÁN", H_HIST)]),
        (["Thuốc trước khi nhập viện", "Thuốc trước khi nhập viện lần này",
          "Bệnh nhân có tiền sử dụng thuốc"],
         [("THUỐC", H_HIST)]),
        (["Tiền sử phẫu thuật / thủ thuật", "Tiền sử phẫu thuật"],
         [("CHẨN_ĐOÁN", H_HIST), ("TÊN_XÉT_NGHIỆM", None)]),
        (["Các tập tương tự trước đây", "Các tập phát bệnh tương tự trước đây",
          "Ghi nhận triệu chứng tương tự trước đây"],
         [("TRIỆU_CHỨNG", H_HIST)]),
    ]),
    (["Tiền sử bệnh hiện tại", "Bệnh sử hiện tại", "Bệnh sử"], [
        (["Lý do nhập viện", "Lý do vào viện"], [("TRIỆU_CHỨNG", None)]),
        (["Triệu chứng khi nhập viện", "Các triệu chứng hiện tại",
          "Triệu chứng hiện tại"], [("TRIỆU_CHỨNG", None)]),
        (["Diễn biến bệnh"], [("TRIỆU_CHỨNG", None), ("CHẨN_ĐOÁN", None)]),
        (["Đặc điểm triệu chứng"], [("TRIỆU_CHỨNG", None)]),
        (["Các sự kiện trước khi nhập viện", "Sự kiện trước khi nhập viện",
          "Tình trạng ngay trước khi nhập viện"],
         [("TRIỆU_CHỨNG", H_HIST), ("TÊN_XÉT_NGHIỆM", None)]),
    ]),
    (["Đánh giá tại bệnh viện", "Khám tại bệnh viện", "Cận lâm sàng"], [
        (["Dấu hiệu lâm sàng", "Toàn trạng"], [("TRIỆU_CHỨNG", None)]),
        (["Kết quả xét nghiệm"],
         [("TÊN_XÉT_NGHIỆM", None), ("KẾT_QUẢ_XÉT_NGHIỆM", None)]),
        (["Kết quả chẩn đoán hình ảnh", "Chẩn đoán hình ảnh"],
         [("KẾT_QUẢ_XÉT_NGHIỆM", None), ("TÊN_XÉT_NGHIỆM", None)]),
        (["Các thủ thuật đã thực hiện"], [("TÊN_XÉT_NGHIỆM", None)]),
        (["Các phát hiện chẩn đoán khác", "Chẩn đoán"], [("CHẨN_ĐOÁN", None)]),
    ]),
    (["Điều trị", "Điều trị tại bệnh viện"], [
        (["Thuốc điều trị", "Thuốc đã dùng"], [("THUỐC", None)]),
    ]),
]

# Dòng nhiễu: KHÔNG BAO GIỜ là concept. Bám đúng những gì test có.
NOISE_KV = [
    ("Vị trí", "Không ghi rõ"),
    ("Thời gian", "Không ghi rõ"),
    ("Mức độ nghiêm trọng", "Không ghi rõ"),
    ("Tần suất", "Không ghi rõ"),
    ("Lan tỏa", "Không ghi rõ"),
    ("Các yếu tố làm nặng thêm", "Không ghi rõ"),
    ("Các yếu tố làm giảm bớt", "Không ghi rõ"),
    ("Khởi phát cấp tính hay từ từ", "Không ghi rõ"),
]
NOISE_LINE = [
    "Không ghi rõ",
    "Không có tiền sử bệnh nào được biết đến",
    "Không rõ đơn",
    "Chưa ghi nhận bất thường",
]

# Khung hỏi–đáp web (35/100 file test có dạng này)
QA_OPEN = [
    "Câu hỏi từ người dùng:", "Câu hỏi từ người dùng :",
    "Câu hỏi của người dùng gửi đến hệ thống", "Hỏi :",
]
QA_ANSWER = [
    "Câu trả lời của bác sĩ:", "Câu trả lời của bác sĩ:  ", "Trả lời:",
]
QA_GREET = [
    "Chào bạn, cảm ơn bạn đã gửi câu hỏi cho chúng tôi.",
    "Chào bạn! Cảm ơn bạn đã gửi câu hỏi cho chúng tôi.",
    "Chào bạn, mình xin trả lời câu hỏi của bạn như sau.",
]

NEG_PREFIX = ["Phủ nhận ", "không ", "không ghi nhận ", "âm tính với "]
BULLETS = ["- ", " - ", "    - ", " • ", "-        "]


def kb_pool(icd=None, rxnorm=None, n_icd: int = 4000, n_rx: int = 1500,
            seed: int = 0, max_len: int = 70) -> dict:
    """Pool concept lấy TỪ CHÍNH CƠ SỞ TRI THỨC (ICD-10 / RxNorm), kèm mã đúng.

    Vì sao: train gold chỉ có **185 mã ICD** và **101 mã RxNorm** phân biệt, trong khi
    `icd10_map.csv` có **13.915** tên bệnh tiếng Việt và `rxnorm_map.csv` có **160.927**
    tên thuốc — đây là nguồn dữ liệu lớn nhất còn chưa dùng để HUẤN LUYỆN (trước giờ chỉ
    dùng lúc tra cứu). Render các tên này thành mention giúp NER mở rộng vốn "trông như
    tên bệnh/tên thuốc" ra ngoài 7.022 chuỗi đã thấy trong train.

    ⚠️ DÙNG CHO NER THÔI. Mọi entry ở đây đều CÓ mã, nên nếu đưa vào train gate thì tỉ lệ
    dương sẽ nhảy từ ~19% lên gần 100% và phá hiệu chuẩn của gate. Notebook phải giữ hai
    luồng tách biệt: `SYNTH` (từ gold) cho NER+assert+gate, `SYNTH_KB` chỉ cho NER.

    THUỐC được lấy nhiều tương đối so với tỉ lệ gold (3.0%) vì train gần như không có mẫu.
    """
    rng = random.Random(seed)
    pool: dict[tuple[str, str | None], list[Concept]] = defaultdict(list)

    def add(typ: str, entries, n: int):
        cands = [e for e in entries if e.description and 3 <= len(e.description) <= max_len]
        if not cands:
            return
        for e in rng.sample(cands, min(n, len(cands))):
            for a in (None, H_HIST, H_NEG):
                pool[(typ, a)].append(Concept(
                    text=e.description, type=typ, position=(0, len(e.description)),
                    assertions=[] if a is None else [a], candidates=[e.code]))

    if icd is not None:
        add("CHẨN_ĐOÁN", icd.entries, n_icd)
    if rxnorm is not None:
        add("THUỐC", rxnorm.entries, n_rx)
    return pool


def merge_pools(*pools: dict) -> dict:
    """Gộp nhiều pool (khóa (type, assertion) -> list Concept)."""
    out: dict[tuple[str, str | None], list[Concept]] = defaultdict(list)
    for p in pools:
        for k, v in p.items():
            out[k].extend(v)
    return out


def build_pool(labeled: list[tuple[str, list[Concept]]]) -> dict:
    """Gom concept gold theo (type, assertion-key) để rút ngẫu nhiên.

    assertion-key: `None` nếu concept không có assertion, ngược lại tên assertion
    (concept có 2 assertion rất hiếm -> lấy assertion đầu theo thứ tự chuẩn).
    """
    pool: dict[tuple[str, str | None], list[Concept]] = defaultdict(list)
    for _text, concepts in labeled:
        for c in concepts:
            a = c.assertions[0] if c.assertions else None
            pool[(c.type, a)].append(c)
    return pool


def _draw(pool: dict, rng: random.Random, typ: str, assertion: str | None):
    cands = pool.get((typ, assertion))
    if not cands:
        return None
    return rng.choice(cands)


def prose_window(labeled, rng: random.Random, target: int = 900):
    """Cắt một CỬA SỔ văn xuôi THẬT (kèm gold) từ train, dóng theo ranh giới dòng.

    Test không phải toàn bullet: 1/3 ký tự là bullet, phần còn lại là văn xuôi
    (hỏi–đáp, bài phổ biến kiến thức, bệnh án kể). Ghép cửa sổ thật vào tài liệu
    tổng hợp cho ra phân phối layout sát test hơn, và văn xuôi vẫn giữ nhãn gold
    nên không dạy model điều sai.
    """
    for _ in range(10):
        text, concepts = rng.choice(labeled)
        if len(text) <= target + 2:
            a, b = 0, len(text)
        else:
            a = rng.randrange(0, len(text) - target)
            a = text.rfind("\n", 0, a) + 1
            b = text.find("\n", a + target)
            b = len(text) if b < 0 else b
        inside = [c for c in concepts if c.position[0] >= a and c.position[1] <= b]
        if inside:
            return text[a:b], [(c, a) for c in inside]
    return "", []


def synth_document(pool: dict, rng: random.Random,
                   qa_prob: float = 0.35, labeled=None,
                   prose_prob: float = 0.65) -> tuple[str, list[Concept]]:
    """Sinh 1 văn bản kiểu test + list Concept có `position` đúng.

    `labeled` (tuỳ chọn): nếu truyền vào, có `prose_prob` xác suất chèn thêm 1–2 cửa
    sổ văn xuôi THẬT kèm gold — mô phỏng việc file test là ghép nhiều đoạn khác thể loại.
    """
    parts: list[str] = []      # các đoạn text
    concepts: list[Concept] = []
    cursor = 0

    def emit(s: str) -> int:
        nonlocal cursor
        parts.append(s)
        start = cursor
        cursor += len(s)
        return start

    def emit_concept(prefix: str, c: Concept, suffix: str = "\n") -> None:
        emit(prefix)
        s = emit(c.text)
        concepts.append(Concept(text=c.text, type=c.type, position=(s, s + len(c.text)),
                                assertions=list(c.assertions),
                                candidates=list(c.candidates)))
        emit(suffix)

    def emit_prose() -> None:
        """Chèn 1 cửa sổ văn xuôi thật, dịch offset gold theo cursor."""
        if not labeled:
            return
        seg, items = prose_window(labeled, rng)
        if not seg:
            return
        base = emit(seg)
        for c, a in items:
            s = base + (c.position[0] - a)
            e = base + (c.position[1] - a)
            concepts.append(Concept(text=c.text, type=c.type, position=(s, e),
                                    assertions=list(c.assertions),
                                    candidates=list(c.candidates)))
        emit("\n")

    qa = rng.random() < qa_prob
    if qa:
        emit(rng.choice(QA_OPEN) + "\n")
        if rng.random() < prose_prob:
            emit_prose()

    n_sec = rng.randint(2, 4)
    # "Điều trị" (toàn THUỐC) chỉ chọn ~1/3 lần: THUỐC chỉ chiếm 3.0% gold train, bơm
    # quá nhiều sẽ đẩy prior sai và `type` sai bị phạt KÉP theo đề (mục 5a).
    catalog = [s for s in SECTIONS[:3]] + ([SECTIONS[3]] if rng.random() < 0.33 else [])
    chosen = rng.sample(catalog, min(n_sec, len(catalog)))
    # giữ thứ tự tự nhiên (tiền sử -> bệnh sử -> đánh giá -> điều trị)
    chosen.sort(key=SECTIONS.index)
    for si, (h1_variants, subs) in enumerate(chosen, start=1):
        emit(f"{si}.  {rng.choice(h1_variants)}\n")
        for h2_variants, slots in rng.sample(subs, rng.randint(1, min(3, len(subs)))):
            emit(f"    {rng.choice(h2_variants)}\n")
            for _ in range(rng.randint(2, 6)):
                if rng.random() < 0.10:
                    if rng.random() < 0.6:
                        k, v = rng.choice(NOISE_KV)
                        emit(f"    - {k}: {v}\n")
                    else:
                        emit(f"    - {rng.choice(NOISE_LINE)}\n")
                    continue
                typ, assertion = rng.choice(slots)
                # đôi khi lấy biến thể phủ định để model thấy cue isNegated trong bullet
                if assertion is None and rng.random() < 0.12:
                    c = _draw(pool, rng, typ, H_NEG)
                    if c is not None:
                        emit_concept(rng.choice(BULLETS) + rng.choice(NEG_PREFIX), c)
                        continue
                c = _draw(pool, rng, typ, assertion)
                if c is None:
                    c = _draw(pool, rng, typ, None)
                if c is None:
                    continue
                emit_concept(rng.choice(BULLETS), c)
        emit("\n")
        if rng.random() < prose_prob * 0.5:
            emit_prose()

    if qa:
        emit(rng.choice(QA_ANSWER) + "\n")
        emit(rng.choice(QA_GREET) + "\n")
        if rng.random() < prose_prob:
            emit_prose()

    text = "".join(parts)
    for c in concepts:
        s, e = c.position
        assert text[s:e] == c.text, "offset lệch khi render"
    return text, concepts


def synth_dataset(labeled: list[tuple[str, list[Concept]]], n: int = 400,
                  seed: int = 0, qa_prob: float = 0.35,
                  splice_prose: bool = True, prose_prob: float = 0.65,
                  pool: dict | None = None
                  ) -> list[tuple[str, list[Concept]]]:
    """Sinh `n` văn bản layout-test từ concept gold của `labeled`.

    `splice_prose=True` (mặc định) ghép thêm cửa sổ văn xuôi thật để phân phối
    bullet/văn xuôi sát tập test (test: ~32% ký tự là bullet).

    `pool` (tuỳ chọn): dùng pool có sẵn thay vì dựng từ `labeled` — vd
    `merge_pools(build_pool(labeled), kb_pool(icd, rxnorm))` để sinh luồng KB cho NER.
    """
    pool = build_pool(labeled) if pool is None else pool
    rng = random.Random(seed)
    src = labeled if splice_prose else None
    return [synth_document(pool, rng, qa_prob=qa_prob, labeled=src,
                           prose_prob=prose_prob) for _ in range(n)]


def to_examples(dataset: list[tuple[str, list[Concept]]]
                ) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """-> [(text, [(start,end,type)])] đúng dạng `ner.encoder.build_features` cần."""
    return [(text, [(c.position[0], c.position[1], c.type) for c in concepts])
            for text, concepts in dataset]


def to_records(dataset: list[tuple[str, list[Concept]]]) -> list[dict]:
    """-> [{'text', 'spans':[[s,e,type]], 'concepts':[dict]}] để ghi JSONL."""
    out = []
    for text, concepts in dataset:
        out.append({
            "text": text,
            "spans": [[c.position[0], c.position[1], c.type] for c in concepts],
            "concepts": [c.to_dict() for c in concepts],
            "source": "layout_synth",
            "split": "train",
        })
    return out
