import os
import fitz

def load_pdf(doc_file):
    doc_bytes = doc_file.read()
    doc = fitz.open(stream=doc_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_docs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
        with open(full_path, "rb") as f:
                text = load_pdf(f)
                documents.append({"file_name": filename, "content": text})
    return documents
