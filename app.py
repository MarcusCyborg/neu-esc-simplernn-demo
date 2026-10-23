"""Giao diện demo; chỉ nạp mô hình khi người dùng bấm Phân tích."""

import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from inference import (
    DEFAULT_ARTIFACT_DIR, LABEL_VI, load_artifacts, load_metrics,
    missing_artifacts, predict_text,
)

st.set_page_config(
    page_title="EduSentiment · Phân tích bình luận",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #16213a;
      --muted: #667085;
      --paper: #f5f7fb;
      --panel: #ffffff;
      --line: #dbe2ee;
      --blue: #3563e9;
      --blue-deep: #2746b8;
      --orange: #f28c28;
      --shadow: 0 18px 45px rgba(29, 52, 96, .10);
    }
    .stApp {
      background:
        radial-gradient(circle at 10% 0%, rgba(242, 140, 40, .13), transparent 30rem),
        radial-gradient(circle at 95% 12%, rgba(53, 99, 233, .12), transparent 32rem),
        var(--paper);
      color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stMainBlockContainer"] {
      max-width: 1120px;
      padding-top: 2.25rem;
      padding-bottom: 4rem;
    }
    .hero {
      position: relative;
      overflow: hidden;
      margin: 0 0 1rem;
      padding: 1.5rem 2rem;
      border: 1px solid rgba(219, 226, 238, .9);
      border-radius: 28px;
      background: linear-gradient(135deg, #1d2d56 0%, #243c7d 58%, #315fd7 100%);
      box-shadow: 0 24px 60px rgba(26, 50, 106, .20);
      color: white;
    }
    .hero::after {
      content: "";
      position: absolute;
      width: 210px;
      height: 210px;
      right: -58px;
      top: -72px;
      border: 28px solid rgba(255, 255, 255, .10);
      border-radius: 50%;
    }
    .hero h1 {
      position: relative;
      z-index: 1;
      max-width: 100%;
      margin: 0;
      color: white;
      /* Segoe UI/Noto Sans include the Vietnamese glyphs used in this heading. */
      font-family: "Segoe UI", "Noto Sans", "Arial Unicode MS", Arial, sans-serif;
      font-size: clamp(1.75rem, 3.5vw, 2.75rem);
      line-height: 1.25;
      letter-spacing: -.025em;
      font-weight: 700;
      white-space: nowrap;
    }
    .hero p {
      position: relative;
      z-index: 1;
      max-width: 800px;
      margin: .65rem 0 0;
      color: rgba(255, 255, 255, .82);
      font-size: .95rem;
      line-height: 1.5;
    }
    .hero-meta {
      position: relative;
      z-index: 1;
      display: flex;
      flex-wrap: wrap;
      gap: .55rem;
      margin-top: .9rem;
    }
    .hero-meta span {
      padding: .42rem .72rem;
      border: 1px solid rgba(255, 255, 255, .2);
      border-radius: 999px;
      background: rgba(255, 255, 255, .10);
      color: rgba(255, 255, 255, .9);
      font-size: .78rem;
      font-weight: 700;
      letter-spacing: .03em;
    }
    [data-testid="stTextArea"] textarea {
      min-height: 75px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, .92);
      color: var(--ink);
      box-shadow: 0 10px 24px rgba(29, 52, 96, .06);
      font-size: 1.06rem;
      line-height: 1.55;
      padding: 1rem 1.05rem;
      transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    [data-testid="stTextArea"] textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 4px rgba(53, 99, 233, .14), 0 12px 28px rgba(29, 52, 96, .08);
      transform: translateY(-1px);
    }
    [data-testid="stTextArea"] label, [data-testid="stSelectbox"] label {
      color: var(--ink);
      font-weight: 800;
      font-size: .9rem;
    }
    [data-testid="stSelectbox"] > div > div {
      border-radius: 13px;
      border-color: var(--line);
      background: rgba(255, 255, 255, .88);
    }
    .hint {
      margin: .45rem 0 0;
      color: var(--muted);
      font-size: .82rem;
    }
    [data-testid="stButton"] button {
      width: 100%;
      min-height: 3.2rem;
      margin-top: .65rem;
      border: 0;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--orange), #ffb04b);
      color: #281505;
      font-size: 1rem;
      font-weight: 900;
      letter-spacing: .01em;
      box-shadow: 0 10px 22px rgba(242, 140, 40, .26);
      transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
    }
    [data-testid="stButton"] button:hover {
      filter: saturate(1.05);
      transform: translateY(-2px);
      box-shadow: 0 14px 27px rgba(242, 140, 40, .34);
    }
    [data-testid="stButton"] button:focus-visible {
      outline: 3px solid rgba(53, 99, 233, .35);
      outline-offset: 3px;
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .7rem;
      margin: 0;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: .5rem;
      padding: .65rem .7rem;
      border: 1px solid rgba(219, 226, 238, .9);
      border-radius: 13px;
      background: rgba(255, 255, 255, .65);
      color: var(--ink);
      font-size: .82rem;
      font-weight: 700;
    }
    .legend-dot { width: .58rem; height: .58rem; border-radius: 50%; flex: 0 0 auto; }
    .st-key-input-panel, .st-key-result-panel {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, .84);
      box-shadow: var(--shadow);
    }
    .st-key-input-panel {
      padding: 1.1rem 1.25rem 1.25rem;
    }
    .st-key-input-panel [data-testid="stMarkdownContainer"]:has(> .legend) {
      margin-bottom: 0;
    }
    .st-key-result-panel {
      margin-top: .25rem;
      padding: 1.1rem 1.25rem .85rem;
      animation: reveal .45s cubic-bezier(.22, 1, .36, 1) both;
    }
    .st-key-result-panel h3 { margin: 0 0 .25rem; color: var(--ink); }
    .result-chip {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      margin-bottom: .5rem;
      padding: .42rem .72rem;
      border-radius: 999px;
      background: #e9eefc;
      color: var(--blue-deep);
      font-size: .8rem;
      font-weight: 800;
    }
    [data-testid="stExpander"] {
      margin-top: 1.5rem;
      border: 1px solid var(--line);
      border-radius: 17px;
      background: rgba(255, 255, 255, .76);
    }
    [data-testid="stMetric"] {
      padding: .85rem .9rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfcff;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stAlert"] { border-radius: 14px; }
    @keyframes reveal {
      from { opacity: 0; transform: translateY(10px); filter: blur(3px); }
      to { opacity: 1; transform: translateY(0); filter: blur(0); }
    }
    @media (max-width: 760px) {
      [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem; }
      .hero { padding: 1.2rem 1.25rem; border-radius: 22px; }
      .hero h1 { font-size: clamp(1.25rem, 4.4vw, 2rem); }
      .legend { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    with st.container(border=True, key="result-panel"):
        st.markdown('<span class="result-chip">● KẾT QUẢ SUY LUẬN</span>', unsafe_allow_html=True)
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
        # DataFrame tạo dataset có tên theo nội dung để biểu đồ cập nhật sau mỗi câu.
        base = alt.Chart(pd.DataFrame(rows)).encode(
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


st.markdown(
    """
    <section class="hero">
      <h1>Phân tích bình luận tiếng Việt</h1>
      <p>EduSentiment dùng SimpleRNN để phân loại bình luận tiếng Việt thành bốn nhãn.</p>
      <div class="hero-meta">
        <span>NEU-ESC</span><span>SimpleRNN</span><span>4 nhãn cảm xúc</span><span>CPU · local demo</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown("### Bạn muốn mô hình đọc câu nào?", unsafe_allow_html=True)
st.markdown('<p class="hint">Nhập trực tiếp hoặc chọn một mẫu ở bên dưới, sau đó bấm phân tích để xem phân bố điểm.</p>', unsafe_allow_html=True)
missing = missing_artifacts()
if missing:
    st.warning("Thiếu artifact: " + ", ".join(missing) + ". Đặt các file vào: " + str(DEFAULT_ARTIFACT_DIR))
with st.container(border=True, key="input-panel"):
    st.text_area("Nhập bình luận", key="comment", height=75,
                 placeholder="Nhập một bình luận tiếng Việt…", on_change=clear_result)
    st.selectbox("Thử một bình luận mẫu", list(EXAMPLES), key="example", on_change=choose_example)
    clicked = st.button("Phân tích", type="primary")
    st.markdown(
        '<div class="legend">'
        '<div class="legend-item"><span class="legend-dot" style="background:#6B7280"></span>Trung tính</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#238443"></span>Tích cực</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#C96813"></span>Tiêu cực</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#C63333"></span>Độc hại</div>'
        '</div>',
        unsafe_allow_html=True,
    )
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
