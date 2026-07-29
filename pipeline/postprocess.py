"""Hậu xử lý prediction trước khi ghi output — các luật RẺ, đo được, bám dữ liệu test.

Mỗi hàm ở đây ra đời từ một quan sát cụ thể trên `data/test`; notebook nên A/B từng
cái chứ đừng bật hết theo niềm tin.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..io.schema import Concept
from ..text.layout import find_headings

_AST_RUN = re.compile(r"\*{3,}")


def drop_redacted(concepts: list[Concept], text: str) -> list[Concept]:
    """Bỏ span chứa dãy ≥3 dấu `*` (tên thuốc đã bị che).

    Quan sát: 30/100 file test có 99 dãy `***` — đúng chỗ tên thuốc bị xoá
    ("Thuốc giảm đau, hạ sốt chứa ******* hoặc **********"). Trong 15.444 concept gold
    của train KHÔNG có dãy `***` nào (chỉ có `*` đơn trong tên xét nghiệm như
    "WBC (Số lượng bạch cầu) *: 6.1 G/L"), nên luật này an toàn: không có gì để map và
    không có gold tương ứng.
    """
    return [c for c in concepts if not _AST_RUN.search(c.text)]


# Heading CẤU TRÚC TÀI LIỆU (không phải khái niệm y tế). Danh sách này là từ vựng tóm
# tắt bệnh án tiếng Việt thông thường, lấy từ khảo sát các dòng heading của `data/test`.
STRUCTURAL_HEADINGS = {
    "tiền sử bệnh", "tiền sử bệnh hiện tại", "tiền sử bệnh nội khoa", "tiền sử bệnh lý",
    "tiền sử bệnh nội", "tiền sử phẫu thuật", "tiền sử phẫu thuật / thủ thuật",
    "bệnh sử hiện tại", "diễn biến bệnh",
    "các bệnh lý mạn tính", "các bệnh lý mãn tính", "các bệnh mãn tính",
    "bệnh lý mãn tính", "bệnh lý mạn tính",
    "lý do nhập viện", "lý do vào viện",
    "triệu chứng hiện tại", "các triệu chứng hiện tại", "triệu chứng khi nhập viện",
    "đặc điểm triệu chứng", "thời điểm khởi phát triệu chứng",
    "triệu chứng liên quan", "các triệu chứng liên quan",
    "các sự kiện trước khi nhập viện", "sự kiện trước khi nhập viện",
    "tình trạng ngay trước khi nhập viện", "thuốc trước khi nhập viện",
    "thuốc trước khi nhập viện lần này",
    "các tập tương tự trước đây", "các tập phát bệnh tương tự trước đây",
    "đánh giá tại bệnh viện", "khám tại bệnh viện", "dấu hiệu lâm sàng", "toàn trạng",
    "cận lâm sàng", "kết quả xét nghiệm", "kết quả chẩn đoán hình ảnh",
    "các thủ thuật đã thực hiện", "các phát hiện chẩn đoán khác",
    "điều trị", "điều trị tại bệnh viện", "thuốc điều trị", "thuốc đã dùng",
    "câu hỏi từ người dùng", "câu trả lời của bác sĩ", "trả lời",
    "câu hỏi của người dùng gửi đến hệ thống", "các yếu tố nguy cơ liên quan",
    "các yếu tố làm nặng thêm", "các yếu tố làm giảm bớt",
    "khởi phát cấp tính hay từ từ", "mức độ nghiêm trọng", "lan tỏa",
    "không ghi rõ", "kết luận", "phòng ngừa", "nguyên nhân", "dinh dưỡng",
}
# Đã LOẠI khỏi danh sách vì chính chúng là gold thật trong train (đếm trên 15.444
# concept): "chẩn đoán hình ảnh" 12 lần (TÊN_XÉT_NGHIỆM), "vận động" 2, "bệnh sử" 2,
# "tần suất" 1, "tiền sử bệnh" 1. Ba cái sau vẫn giữ lại vì chỉ 1-2 ca lẻ trong khi
# test có hàng trăm dòng heading — nhưng "chẩn đoán hình ảnh" thì rõ ràng phải giữ.


def _norm_head(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(":.-•* \t").lower()


def drop_headings(concepts: list[Concept], text: str) -> list[Concept]:
    """Bỏ span mà nội dung LÀ một heading cấu trúc tài liệu.

    Trong test có 33 dòng "Đánh giá tại bệnh viện", 33 "Tiền sử bệnh hiện tại",
    30 "Tiền sử bệnh"… nếu NER tag chúng thì vừa mất precision vừa phình union Jaccard.

    Chỉ dùng danh sách CỐ ĐỊNH `STRUCTURAL_HEADINGS`, KHÔNG dùng bộ nhận diện heading
    tự động: bản tự động xoá mất 0.47% gold thật của train (vd "Đo hoạt độ Lipase",
    "Điện giải đồ (Na, K, Cl)" đứng trọn một dòng nên bị nhận nhầm là heading).
    Với danh sách cố định, thiệt hại trên train là 1/15.444 concept ("tiền sử bệnh").
    """
    return [c for c in concepts if _norm_head(c.text) not in STRUCTURAL_HEADINGS]


def unify_types(concepts: list[Concept]) -> list[Concept]:
    """Cùng một chuỗi trong CÙNG file -> thống nhất `type` theo số nhiều.

    Đề (mục 5a) phạt KÉP khi sai `type`: concept bị tách thành 2 và cả 2 đều 0 điểm ở
    cả 3 metric. Gold trong 1 file thường nhất quán (vd test/53 có "gầy sút cân" 2 lần,
    cùng TRIỆU_CHỨNG), còn NER thì hay lung lay giữa TRIỆU_CHỨNG và CHẨN_ĐOÁN — đúng
    nhầm lẫn đã biết từ baseline. Bỏ phiếu theo chuỗi giúp giảm phần lung lay đó.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for c in concepts:
        votes[c.text.strip().lower()][c.type] += 1
    for c in concepts:
        best = votes[c.text.strip().lower()].most_common(1)
        if best:
            c.type = best[0][0]
    return concepts


def unify_labels(concepts: list[Concept]) -> list[Concept]:
    """Cùng chuỗi + cùng type trong 1 file -> thống nhất `candidates` (đa số, ưu tiên có mã).

    `assertions` KHÔNG thống nhất: cùng một triệu chứng có thể vừa là tiền sử ở mục
    "Tiền sử bệnh" vừa là hiện tại ở mục "Triệu chứng hiện tại" — thấy rõ trong test.
    """
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for c in concepts:
        if c.candidates:
            votes[(c.text.strip().lower(), c.type)][tuple(c.candidates)] += 1
    for c in concepts:
        v = votes.get((c.text.strip().lower(), c.type))
        if v:
            c.candidates = list(v.most_common(1)[0][0])
    return concepts


def apply(concepts: list[Concept], text: str, redacted: bool = True,
          headings: bool = True, types: bool = True, labels: bool = True
          ) -> list[Concept]:
    """Chạy các bước bật sẵn, theo thứ tự an toàn (lọc trước, thống nhất sau)."""
    if redacted:
        concepts = drop_redacted(concepts, text)
    if headings:
        concepts = drop_headings(concepts, text)
    if types:
        concepts = unify_types(concepts)
    if labels:
        concepts = unify_labels(concepts)
    return concepts
