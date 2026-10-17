import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# -------------------- Cấu hình trang --------------------
st.set_page_config(
    page_title="AI Cố vấn học tập",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Giao diện --------------------
st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #0b3b73, #176b87);
        color: white;
        margin-bottom: 1rem;
      }
      .hero h1 { color: white; margin: 0 0 .45rem 0; font-size: 2rem; }
      .hero p { color: #e8f3ff; margin: 0; font-size: 1rem; }
      .info-card {
        background: white; border: 1px solid #e5eaf2; border-radius: 16px;
        padding: 1rem 1.1rem; margin-bottom: .7rem;
      }
      .small-muted { color: #5f6b7a; font-size: .88rem; }
      div[data-testid="stChatMessage"] {
        background: white; border: 1px solid #e7ebf2; border-radius: 14px;
        padding: .5rem .8rem;
      }
      .stButton > button {
        border-radius: 12px; border: 1px solid #d8e1ed;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Đọc dữ liệu --------------------
DATA_FILE = Path(__file__).parent / "knowledge_base.xlsx"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.40


def find_column(columns, keyword):
    """Tìm tên cột có chứa từ khóa, không phân biệt chữ hoa/thường."""
    for col in columns:
        if keyword in str(col).strip().lower():
            return col
    return None


@st.cache_data
def load_knowledge_base(file_path: str):
    df = pd.read_excel(file_path)
    question_col = find_column(df.columns, "câu hỏi")
    answer_col = find_column(df.columns, "câu trả lời")

    # Nếu tên cột trong Excel khác dự kiến, thử tìm theo tiếng Anh.
    if question_col is None:
        question_col = find_column(df.columns, "question")
    if answer_col is None:
        answer_col = find_column(df.columns, "answer")

    if question_col is None or answer_col is None:
        raise ValueError(
            "Không tìm thấy cột câu hỏi/câu trả lời. "
            f"Các cột đang có: {df.columns.tolist()}"
        )

    clean = df[[question_col, answer_col]].dropna().copy()
    clean[question_col] = clean[question_col].astype(str).str.strip()
    clean[answer_col] = clean[answer_col].astype(str).str.strip()
    clean = clean[
        (clean[question_col] != "") & (clean[answer_col] != "")
    ].reset_index(drop=True)

    return clean[question_col].tolist(), clean[answer_col].tolist()


@st.cache_resource(show_spinner="Đang tải mô hình AI lần đầu, bạn vui lòng đợi...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


def get_answer(user_question, questions, answers, model, question_vectors):
    user_question = user_question.strip()
    if not user_question:
        return "Bạn hãy nhập câu hỏi để mình hỗ trợ nhé.", None, None

    user_vector = model.encode(
        [user_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = question_vectors @ user_vector
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return (
            "Mình chưa tìm thấy thông tin đủ phù hợp trong bộ câu hỏi hiện có. "
            "Bạn hãy thử diễn đạt cụ thể hơn hoặc xác nhận với đơn vị phụ trách học vụ "
            "của Học viện trước khi thực hiện các thủ tục quan trọng.",
            questions[best_idx],
            best_score,
        )

    return answers[best_idx], questions[best_idx], best_score


# -------------------- Phần đầu trang --------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 AI Cố vấn học tập</h1>
      <p>Trợ lý hỏi–đáp học vụ dành cho sinh viên · Tra cứu từ bộ câu hỏi thường gặp</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎓 Trợ lý học tập")
    st.write("Đặt câu hỏi bằng tiếng Việt về các nội dung có trong bộ dữ liệu.")
    st.markdown("---")
    st.markdown("### 💡 Gợi ý câu hỏi")
    examples = [
        "Hồ sơ nhập học gồm những giấy tờ gì?",
        "Em muốn đăng ký ký túc xá thì làm thế nào?",
        "Có chứng chỉ tiếng Anh thì có được miễn học không?",
        "Khi nào đăng ký học phần?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["suggested_question"] = example
    st.markdown("---")
    st.caption(
        "Lưu ý: Đây là dự án hỗ trợ tra cứu độc lập, không phải website chính thức "
        "của Học viện. Hãy xác minh quy định và thời hạn trên thông báo chính thức."
    )

# -------------------- Nạp dữ liệu và mô hình --------------------
if not DATA_FILE.exists():
    st.error("Không tìm thấy knowledge_base.xlsx. Hãy đặt file Excel cùng thư mục với app.py.")
    st.stop()

try:
    questions, answers = load_knowledge_base(str(DATA_FILE))
except Exception as exc:
    st.error(f"Không đọc được file dữ liệu: {exc}")
    st.stop()

if not questions:
    st.error("File Excel chưa có cặp câu hỏi và câu trả lời hợp lệ.")
    st.stop()

st.markdown(
    f"""
    <div class="info-card">
      <strong>📚 Cơ sở dữ liệu</strong><br>
      <span class="small-muted">Đã nạp {len(questions)} cặp câu hỏi – câu trả lời.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Đang chuẩn bị trợ lý AI..."):
    try:
        model = load_model()
        question_vectors = model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        st.error(
            "Không tải được mô hình AI. Hãy kiểm tra kết nối hoặc thử chạy lại ứng dụng. "
            f"Chi tiết: {exc}"
        )
        st.stop()

# -------------------- Hội thoại --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! 👋 Mình là AI Cố vấn học tập. "
                "Bạn có thể hỏi mình về các thông tin có trong bộ câu hỏi thường gặp."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

suggested = st.session_state.pop("suggested_question", "")
prompt = st.chat_input("Nhập câu hỏi của bạn tại đây...")

if suggested and not prompt:
    prompt = suggested

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm thông tin phù hợp..."):
            response, matched_question, score = get_answer(
                prompt, questions, answers, model, question_vectors
            )
        st.markdown(response)
        if matched_question is not None:
            with st.expander("Xem câu hỏi trong dữ liệu được dùng để đối chiếu"):
                st.write(matched_question)
                st.caption(f"Độ tương đồng tham khảo: {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("🗑️ Xóa hội thoại"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hội thoại đã được làm mới. Bạn muốn tra cứu thông tin gì?",
        }
    ]
    st.rerun()
