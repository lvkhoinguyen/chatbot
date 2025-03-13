import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import time
import spacy  # Add this import

# Cấu hình API key của Gemini
genai.configure(api_key="api")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Hàm trích xuất nội dung từ file PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text
    return extracted_text

# Chia văn bản thành các đoạn nhỏ
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])  # Fixed the indexing here
    return chunks

# Tạo embeddings cho các đoạn văn bản và tìm đoạn liên quan bằng FAISS
def find_relevant_chunks(question, chunks, model, top_n=3):
    # Tạo embeddings
    chunk_embeddings = model.encode(chunks)
    question_embedding = model.encode([question])

    # Xây dựng FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    # Tìm kiếm top N đoạn liên quan
    _, top_indices = index.search(question_embedding, top_n)
    relevant_chunks = [chunks[i] for i in top_indices[0]]
    return relevant_chunks

# Hàm trích xuất thực thể từ văn bản
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Hỏi mô hình Gemini với ngữ cảnh
def ask_gemini(question, context=None, history=None):
    history_text = ""
    if history:
        for q, a in history:
            history_text += f"Q: {q}\nA: {a}\n"

    prompt = f"""
    Bạn là một trợ lý hữu ích. Trả lời câu hỏi dựa trên ngữ cảnh và lịch sử hội thoại đã cung cấp.
    Nếu ngữ cảnh không chứa thông tin cần thiết để trả lời câu hỏi, hãy nói "Tôi không biết."

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

    return thought_process, final_answer

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

            # Sử dụng mô hình Sentence Transformers để tạo embeddings
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            relevant_chunks = find_relevant_chunks(question, all_chunks, embedding_model)
            combined_context = " ".join(relevant_chunks)

            # Show "Thinking..." indicator
            st.info("Gemini đang suy nghĩ...")

            # Hỏi mô hình Gemini với suy nghĩ
            thought_process, final_answer = ask_gemini_with_thoughts(question, combined_context, st.session_state.conversation_history)

            # Lưu câu hỏi và câu trả lời vào lịch sử
            st.session_state.conversation_history.append((question, final_answer))

            # Hiển thị suy nghĩ và câu trả lời
            st.markdown(f"<div style='background-color: #fff; padding: 15px; border-radius: 10px;'><b>Suy nghĩ của Gemini:</b><br>{thought_process}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #fff; padding: 15px; border-radius: 10px;'><b>Trả lời:</b><br>{final_answer}</div>",
                        unsafe_allow_html=True)

            # Extract and display named entities
            entities = extract_entities(final_answer)
            if entities:
                st.markdown("<b>Thực thể được nhận dạng:</b>", unsafe_allow_html=True)
                for entity, label in entities:
                    st.markdown(f"- **{entity}**: {label}", unsafe_allow_html=True)

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