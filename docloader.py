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

# def load_docs_from_folder(folder_path):
#     documents = ""
