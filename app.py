"""Giao diện demo; chỉ nạp mô hình khi người dùng bấm Phân tích."""

import json
import math
from pathlib import Path

import altair as alt
import streamlit as st

from inference import (
    DEFAULT_ARTIFACT_DIR, LABEL_VI, load_artifacts, load_metrics,
    missing_artifacts, predict_text,
)

st.set_page_config(page_title="Phân tích bình luận tiếng Việt", layout="centered")

EXAMPLES = {
    "Tự nhập bình luận": "",
    "Một lời khen": "Bài viết rất hữu ích, cảm ơn bạn đã chia sẻ!",
    "Một lời phàn nàn": "Dịch vụ quá chậm, tôi rất thất vọng.",
    "Một thông báo": "Buổi học bắt đầu lúc 8 giờ sáng mai.",
    "Một bình luận công kích": "Đồ ngu, đừng lên đây nói linh tinh nữa.",
}
COLORS = ["#6B7280", "#238443", "#C96813", "#C63333"]


@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


def clear_result():
    st.session_state.pop("result", None)


def choose_example():
    st.session_state["comment"] = EXAMPLES[st.session_state["example"]]
    clear_result()


def read_config():
    # Đọc JSON nhỏ để phần thông tin vẫn hoạt động khi chưa nạp TensorFlow.
    try:
        config = json.loads((DEFAULT_ARTIFACT_DIR / "config.json").read_text(encoding="utf-8"))
        return config if isinstance(config, dict) else {}
    except (OSError, ValueError):
        return {}


def show_scores(result):
    st.subheader(f"Nhãn dự đoán: {result['label']}")
    if result.get("truncated"):
        st.warning(
            f"Bình luận có {result['token_count']} token sau chuẩn hóa. "
            f"Mô hình chỉ sử dụng {result['max_length']} token đầu tiên."
        )
    st.markdown("#### Điểm dự đoán của mô hình")
    rows = [
        {"Nhãn": label, "Điểm": score, "Hiển thị": f"{score:.1%}"}
        for label, score in zip(LABEL_VI, result["scores"])
    ]
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("Nhãn:N", sort=list(LABEL_VI), title=None),
        x=alt.X("Điểm:Q", scale=alt.Scale(domain=[0, 1.15]),
                  axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1]), title=None),
    )
    bars = base.mark_bar(size=25).encode(
        color=alt.Color("Nhãn:N", scale=alt.Scale(domain=list(LABEL_VI), range=COLORS), legend=None),
        tooltip=["Nhãn:N", alt.Tooltip("Điểm:Q", format=".2%")],
    )
    labels = base.mark_text(align="left", dx=7, color="#172033", fontSize=14).encode(text="Hiển thị:N")
    st.altair_chart((bars + labels).properties(height=205).configure_view(stroke=None), width="stretch")
    st.caption("Điểm softmax không phải độ chính xác được bảo đảm cho từng câu.")


def show_information():
    with st.expander("Thông tin mô hình và đánh giá"):
        st.markdown(
            "**Dữ liệu:** [NEU-ESC](https://huggingface.co/datasets/hung20gg/NEU-ESC). "
            "Mô hình được huấn luyện và đánh giá trên các tập con phân tầng; "
            "kết quả dưới đây thuộc tập con test, không đại diện cho toàn bộ test NEU-ESC."
        )
        sizes = read_config().get("subset_sizes")
        if isinstance(sizes, dict):
            st.write("Số mẫu trong từng tập:", sizes)
        st.markdown("**Kiến trúc**")
        st.code("Embedding(64, mask_zero=True)\n→ SimpleRNN(64, tanh, return_sequences=True)\n→ GlobalAveragePooling1D (có mask)\n→ Dropout(0.2)\n→ Dense(4, softmax)", language=None)
        metrics = load_metrics()
        if metrics is None:
            st.info("Chưa có kết quả đánh giá")
        else:
            test = metrics["test"]
            columns = st.columns(3)
            for column, name, key in zip(columns, ["Accuracy", "Macro-F1", "F1 Toxic"], ["accuracy", "macro_f1", "f1_toxic"]):
                column.metric(name, f"{test[key]:.3f}")
            baseline = metrics.get("baseline_test")
            if isinstance(baseline, dict) and all(
                isinstance(baseline.get(k), (int, float)) and math.isfinite(baseline[k]) and 0 <= baseline[k] <= 1
                for k in ("accuracy", "macro_f1", "f1_toxic")
            ):
                st.caption(f"Baseline luôn chọn lớp đông nhất: Accuracy {baseline['accuracy']:.3f}, Macro-F1 {baseline['macro_f1']:.3f}, F1 Toxic {baseline['f1_toxic']:.3f}.")
            for filename, caption in [("confusion_matrix.png", "Ma trận nhầm lẫn trên tập con test"), ("training_curves.png", "Đường cong huấn luyện")]:
                path = DEFAULT_ARTIFACT_DIR / filename
                if path.is_file():
                    try:
                        from PIL import Image
                        with Image.open(path) as image:
                            image.load()
                            st.image(image, caption=caption, width="stretch")
                    except Exception:
                        st.caption(f"Không thể hiển thị {filename}.")
        st.markdown("**Giới hạn:** Câu dài bị cắt theo cấu hình. Mỉa mai, tiếng lóng, ngữ cảnh thiếu và lớp ít mẫu (đặc biệt Toxic) có thể gây dự đoán sai. Kết quả chỉ phục vụ minh họa mô hình.")


st.title("Phân tích bình luận tiếng Việt")
st.write("Phân loại 4 nhãn với mô hình SimpleRNN được huấn luyện trên NEU-ESC.")
missing = missing_artifacts()
if missing:
    st.warning("Thiếu artifact: " + ", ".join(missing) + ". Đặt các file vào: " + str(DEFAULT_ARTIFACT_DIR))
st.text_area("Nhập bình luận", key="comment", height=140,
             placeholder="Nhập một bình luận tiếng Việt…", on_change=clear_result)
st.selectbox("Thử một bình luận mẫu", list(EXAMPLES), key="example", on_change=choose_example)
st.caption("Các câu mẫu chỉ để thử giao diện; nhãn được xác định bởi mô hình khi bấm Phân tích.")
clicked = st.button("Phân tích", type="primary")
text = st.session_state.get("comment", "")
if clicked:
    clear_result()
    if not text.strip():
        st.info("Vui lòng nhập bình luận trước khi phân tích.")
    else:
        missing = missing_artifacts()
        if missing:
            st.warning("Chưa có đủ tệp mô hình. Hãy đặt các tệp còn thiếu vào thư mục artifacts rồi thử lại.")
            st.code(str(DEFAULT_ARTIFACT_DIR) + "\n\n" + "\n".join(missing), language=None)
        else:
            try:
                with st.spinner("Đang phân tích bình luận…"):
                    st.session_state["result"] = predict_text(text, cached_artifacts())
            except Exception as error:
                st.error("Không thể phân tích bình luận. Hãy kiểm tra bộ artifact và môi trường cài đặt rồi thử lại.")
                with st.expander("Chi tiết kỹ thuật"):
                    st.exception(error)
result = st.session_state.get("result")
if result and result.get("text") == text:
    show_scores(result)
elif result:
    clear_result()

show_information()
