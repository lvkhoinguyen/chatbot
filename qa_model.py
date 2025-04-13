from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Đường dẫn lưu trữ cache
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Hàm lưu cache
def save_cache(file_name, data):
    with open(os.path.join(CACHE_DIR, file_name), "wb") as f:
        pickle.dump(data, f)

# Hàm tải cache
def load_cache(file_name):
    file_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

# Hàm tạo embedding với caching
def create_embeddings_with_cache(chunks, model_name='all-MiniLM-L6-v2'):
    cache_file = f"embeddings_{model_name}.pkl"
    cached_data = load_cache(cache_file)

    if cached_data:
        print("Sử dụng embedding từ cache.")
        return cached_data

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    save_cache(cache_file, (embeddings, model))
    return embeddings, model

# Hàm tìm kiếm với caching
def find_relevant_chunks_with_cache(question, chunks, embeddings, model, top_n=3):
    cache_file = f"query_{hash(question)}.pkl"
    cached_result = load_cache(cache_file)

    if cached_result:
        print("Sử dụng kết quả truy vấn từ cache.")
        return cached_result

    relevant_chunks = find_relevant_chunks(question, chunks, embeddings, model, top_n)
    save_cache(cache_file, relevant_chunks)
    return relevant_chunks

# Hàm tạo embedding cho các đoạn văn
def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# Hàm tìm kiếm các đoạn liên quan nhất
def find_relevant_chunks(question, chunks, embeddings, model, top_n=3):
    # Tạo embedding cho câu hỏi
    question_embedding = model.encode([question])

    # Xây dựng FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Tìm kiếm top N đoạn liên quan
    _, top_indices = index.search(question_embedding, top_n)
    relevant_chunks = [chunks[i] for i in top_indices[0]]
    return relevant_chunks

def re_rank_chunks(question, chunks, model_name='all-MiniLM-L6-v2'):
    """
    Re-rank the retrieved chunks based on semantic relevance to the question.

    Args:
        question (str): The user query.
        chunks (list): List of text chunks retrieved from the initial search.
        model_name (str): The name of the SentenceTransformer model to use.

    Returns:
        list: A list of tuples (chunk, score) sorted by relevance score in descending order.
    """
    from sentence_transformers import SentenceTransformer, util

    # Load the model
    model = SentenceTransformer(model_name)

    # Encode the question and chunks
    question_embedding = model.encode(question, convert_to_tensor=True)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    # Compute cosine similarity scores
    scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]

    # Pair each chunk with its score
    scored_chunks = list(zip(chunks, scores.tolist()))

    # Sort chunks by score in descending order
    scored_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

    return scored_chunks