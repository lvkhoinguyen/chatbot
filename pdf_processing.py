import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
    extracted_text = ""
    for page in pdf_reader:
        text = page.get_text()  # Trích xuất văn bản từ trang
        if text:
            extracted_text += text
    pdf_reader.close()
    return extracted_text