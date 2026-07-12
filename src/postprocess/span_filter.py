from __future__ import annotations

from src.io.schema import SpanPrediction


GENERIC_DIAGNOSIS_TEXTS = {
    "bệnh",
    "bệnh hiện tại",
    "tiền sử bệnh",
    "chẩn đoán",
    "hình ảnh",
    "khác",
    "không xác định",
}
LOW_VALUE_DIAGNOSIS_PREFIXES = (
    "phẫu thuật",
    "thủ thuật",
    "xử trí",
    "lý do nhập viện",
    "hình ảnh",
    "thăm dò",
    "sau đó",
    "trước vào viện",
    "hiện ",
)
NON_SYMPTOM_PREFIXES = ("khi ", "lúc ", "tại ")
NON_SYMPTOM_WORDS = ("khoa", "bệnh viện", "cấp cứu")
LOW_VALUE_LAB_TEST_TEXTS = {
    "dấu hiệu sinh tồn là",
    "được con trai phát hiện tại nhà lúc khoảng 11",
    "khoảng 11",
    "lý do nhập viện",
    "nhịp thở",
    "spo2",
    "tần suất",
    "thời gian",
    "thời gian khởi phát",
    "thời điểm khởi phát triệu chứng",
    "truyền dịch",
}
LOW_VALUE_LAB_TEST_PREFIXES = (
    "bắt đầu nhận thấy",
    "được con trai phát hiện",
    "thời điểm khởi phát",
)
LOW_VALUE_SYMPTOM_TEXTS = {
    "các biến trước khi nhập viện",
    "các diễn biến trước khi nhập viện",
    "cận lâm sàng trước khi nhập viện",
    "cần phải ho nó ra",
    "đã khám bác sĩ chăm sóc chính hai lần",
    "đến gặp bác sĩ chăm sóc chính sáng nay",
    "đi khám lại bác sĩ chăm sóc chính vào hôm qua",
    "điều trị",
    "ho rối loạn cảm xúc (trầm cảm) (tiền sử giai đoạn trầm cảm không đặc hiệu)",
    "hoặc các triệu chứng khác",
    "nhập viện tại khoa cấp cứu để điều trị thêm",
    "ngày càng nặng hơn",
    "sau đó nhaoaj viện tại khoa cấp cứu",
    "sự kiện trước khi nhập viện",
    "thỉnh thoảng tiêu chảy các thuốc của ông ấy",
    "thủ thuật điện chọn",
    "tổn thương chi dưới do tự tử không thành",
    "vị trí không xác định",
    "xảy ra vài lần mỗi ngày",
}
LOW_VALUE_SYMPTOM_PREFIXES = (
    "cơn đau thắt ngực không ổn định cần ",
    "đến khám để thực hiện ",
    "nhập viện tại",
    "phim chụp ct hình ảnh ",
    "sau đó nhaoaj viện",
)


def filter_low_value_spans(spans: list[SpanPrediction]) -> list[SpanPrediction]:
    filtered: list[SpanPrediction] = []
    for span in spans:
        text_norm = " ".join(span.text.lower().strip(" :-").split())
        if span.concept_type == "symptom":
            if text_norm in LOW_VALUE_SYMPTOM_TEXTS:
                continue
            if text_norm.startswith(LOW_VALUE_SYMPTOM_PREFIXES):
                continue
            if "khoa cấp cứu" in text_norm and "nhập viện" in text_norm:
                continue
        if span.concept_type == "lab_test":
            if text_norm.startswith("nội soi (") and any(marker in text_norm for marker in ("cách đây", "cho thấy", "phát hiện")):
                continue
        if span.source != "model_exact":
            if span.concept_type == "diagnosis":
                if text_norm in GENERIC_DIAGNOSIS_TEXTS:
                    continue
                if ":" in span.text and text_norm.startswith(("hình ảnh", "lý do nhập viện", "kết quả chẩn đoán")):
                    continue
                if not span.candidates:
                    if span.source == "diagnosis_cue" or len(span.text) > 45:
                        continue
                    if text_norm.startswith(LOW_VALUE_DIAGNOSIS_PREFIXES):
                        continue
            if span.concept_type == "lab_test" and len(span.text.strip()) == 1:
                continue
            if span.concept_type == "lab_test":
                if text_norm in LOW_VALUE_LAB_TEST_TEXTS:
                    continue
                if text_norm.startswith(LOW_VALUE_LAB_TEST_PREFIXES):
                    continue
            if span.concept_type == "symptom":
                if text_norm.startswith(NON_SYMPTOM_PREFIXES) and any(word in text_norm for word in NON_SYMPTOM_WORDS):
                    continue
        filtered.append(span)
    return filtered
