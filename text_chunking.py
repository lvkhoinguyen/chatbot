import re

def split_text_into_chunks(text, min_tokens=200, max_tokens=500):
    # Tách văn bản thành các đoạn dựa trên tiêu đề hoặc đoạn văn
    paragraphs = re.split(r'\n\s*\n', text)  # Tách theo dòng trống
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk.split()) + len(paragraph.split()) <= max_tokens:
            current_chunk += paragraph + "\n"
        else:
            if len(current_chunk.split()) >= min_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks