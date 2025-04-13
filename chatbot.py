import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import time
import spacy  # Add this import
import fitz  # PyMuPDF
import re

from pdf_processing import extract_text_from_pdf
from text_chunking import split_text_into_chunks
from qa_model import create_embeddings_with_cache, find_relevant_chunks_with_cache, re_rank_chunks
from entity_extraction import extract_entities

# Cấu hình API key của Gemini
genai.configure(api_key="api")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Hỏi mô hình Gemini với ngữ cảnh
def ask_gemini(question, context=None, history=None):
    history_text = ""
    if history:
        for q, a in history:
            history_text += f"Q: {q}\nA: {a}\n"

    prompt = f"""
    Bạn là một trợ lý hữu ích. Trả lời câu hỏi dựa trên ngữ cảnh và lịch sử hội thoại đã cung cấp.
    "

    Lịch sử hội thoại:
    {history_text}

    Ngữ cảnh: {context}

    Câu hỏi: {question}
    """
    
    # Simulate thinking by adding a delay
    time.sleep(2)  # 2-second delay to simulate thinking
    
    response = model.generate_content(prompt)
    return response.text

# Hỏi mô hình Gemini với suy nghĩ
def ask_gemini_with_thoughts(question, context=None, history=None):
    history_text = ""
    if history:
        for q, a in history:
            history_text += f"Q: {q}\nA: {a}\n"

    thought_prompt = f"""
    Bạn là một trợ lý hữu ích. Trước khi trả lời câu hỏi, hãy suy nghĩ và giải thích quá trình suy nghĩ của bạn.
    Nếu ngữ cảnh không chứa thông tin cần thiết để trả lời câu hỏi, hãy nói "Tôi không biết."

    Lịch sử hội thoại:
    {history_text}

    Ngữ cảnh: {context}

    Câu hỏi: {question}

    Suy nghĩ của tôi:
    """
    
    # Simulate thinking by adding a delay
    time.sleep(2)  # 2-second delay to simulate thinking
    
    thought_response = model.generate_content(thought_prompt)
    thought_process = thought_response.text

    final_prompt = f"""
    Bạn là một trợ lý hữu ích. Dựa trên quá trình suy nghĩ sau đây, hãy trả lời câu hỏi.

    Suy nghĩ của tôi:
    {thought_process}

    Câu trả lời:
    """
    
    final_response = model.generate_content(final_prompt)
    final_answer = final_response.text

    # Check if the answer contains "Tôi không biết"
    if "Tôi không biết" in final_answer:
        # Ask Gemini directly without context
        direct_prompt = f"""
        Bạn là một trợ lý hữu ích. Trả lời câu hỏi sau mà không cần ngữ cảnh.

        Câu hỏi: {question}
        """
        direct_response = model.generate_content(direct_prompt)
        final_answer = direct_response.text

    return thought_process, final_answer

# Hỏi mô hình GPT với prompt được định dạng hợp lý
def ask_gpt_with_prompt(question, relevant_chunks, history=None):
    """
    Gửi câu hỏi và các đoạn văn bản liên quan đến GPT với prompt được định dạng hợp lý.

    Args:
        question (str): Câu hỏi của người dùng.
        relevant_chunks (list): Các đoạn văn bản liên quan.
        history (list): Lịch sử hội thoại (nếu có).

    Returns:
        str: Câu trả lời từ GPT.
    """
    history_text = ""
    if history:
        for q, a in history:
            history_text += f"Q: {q}\nA: {a}\n"

    # Định dạng prompt
    context_text = "\n\n".join([f"[Đoạn {i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
    prompt = f"""
    Dưới đây là các đoạn thông tin trích từ tài liệu:
    {context_text}

    Câu hỏi: {question}
    Trả lời dựa trên nội dung trên:
    Nếu không có thông tin trong tài liệu, hãy trả lời là 'Tôi không chắc chắn'.
    """

    # Gửi prompt đến GPT
    response = model.generate_content(prompt)
    return response.text

# Hỏi mô hình GPT với ngữ cảnh hội thoại

def ask_gpt_with_context(question, relevant_chunks, history=None):
    """
    Gửi câu hỏi và các đoạn văn bản liên quan đến GPT với ngữ cảnh hội thoại.

    Args:
        question (str): Câu hỏi của người dùng.
        relevant_chunks (list): Các đoạn văn bản liên quan.
        history (list): Lịch sử hội thoại (nếu có).

    Returns:
        str: Câu trả lời từ GPT.
    """
    # Định dạng lịch sử hội thoại
    history_text = ""
    if history:
        for i, (q, a) in enumerate(history):
            history_text += f"Turn {i+1}:\nQ: {q}\nA: {a}\n\n"

    # Định dạng prompt
    context_text = "\n\n".join([f"[Đoạn {i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
    prompt = f"""
    Dưới đây là các đoạn thông tin trích từ tài liệu:
    {context_text}

    Lịch sử hội thoại:
    {history_text}

    Câu hỏi hiện tại: {question}
    Trả lời dựa trên nội dung trên và lịch sử hội thoại:
    Nếu không có thông tin trong tài liệu, hãy trả lời là 'Tôi không chắc chắn'.
    """

    # Gửi prompt đến GPT
    response = model.generate_content(prompt)
    return response.text

# Giao diện Streamlit
st.set_page_config(page_title="Hỏi Đáp với Gemini - RAG", layout="wide")

st.title("Hỏi Đáp với Gemini (RAG)")

# Lưu trữ nội dung PDF trong session
if "pdf_contents" not in st.session_state:
    st.session_state.pdf_contents = {}

# Lưu lịch sử hội thoại
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Cho phép tải lên nhiều file PDF
uploaded_files = st.file_uploader("Tải lên các file PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.pdf_contents:
            with st.spinner(f"Đang xử lý {file_name}..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                st.session_state.pdf_contents[file_name] = split_text_into_chunks(pdf_text)

    st.success("Các file PDF đã được tải lên và xử lý thành công!")

# Hiển thị danh sách các file PDF đã tải lên
if st.session_state.pdf_contents:
    st.subheader("Danh sách file PDF đã tải lên:")
    for file_name in st.session_state.pdf_contents.keys():
        st.write(f"- {file_name}")

# Nhập câu hỏi và tìm câu trả lời
question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu?")

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Hỏi Gemini"):
        with st.spinner("Đang tìm câu trả lời..."):
            all_chunks = []
            for chunks in st.session_state.pdf_contents.values():
                all_chunks.extend(chunks)

            # Tạo embedding cho các đoạn văn
            embedding_model_name = 'all-MiniLM-L6-v2'
            embeddings, embedding_model = create_embeddings_with_cache(all_chunks, embedding_model_name)

            # Tìm các đoạn liên quan nhất
            relevant_chunks = find_relevant_chunks_with_cache(question, all_chunks, embeddings, embedding_model)
            combined_context = " ".join(relevant_chunks)

            # Gửi câu hỏi và ngữ cảnh vào GPT để tạo câu trả lời
            final_answer = ask_gemini(question, combined_context, st.session_state.conversation_history)

            # Lưu câu hỏi và câu trả lời vào lịch sử
            st.session_state.conversation_history.append((question, final_answer))

            # Hiển thị câu trả lời
            st.markdown(f"<div style='background-color: #fff; padding: 15px; border-radius: 10px;'><b>Trả lời:</b><br>{final_answer}</div>",
                        unsafe_allow_html=True)

with col2:
    if st.checkbox("Hiển thị nội dung các file PDF"):
        for file_name, chunks in st.session_state.pdf_contents.items():
            with st.expander(f"Nội dung của {file_name}"):
                st.text_area("", "\n".join(chunks), height=300, key=f"pdf_{file_name}")

# Hiển thị lịch sử hội thoại
if st.session_state.conversation_history:
    st.subheader("Lịch sử hội thoại")
    for i, (q, a) in enumerate(st.session_state.conversation_history):
        st.markdown(f"**Câu hỏi {i+1}:** {q}")
        st.markdown(f"**Trả lời:** {a}")