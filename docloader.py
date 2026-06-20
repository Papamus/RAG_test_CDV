import os
import fitz

def load_pdf(doc_file):
    doc_file.seek(0)
    # ustawiam kursor na początek pliku, aby upewnić się, że odczytujemy cały plik PDF

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
            full_path = os.path.join(folder_path, filename) # Definicja full_path
            with open(full_path, "rb") as f:
                text = load_pdf(f)
            documents.append({"file_name": filename, "content": text})
    return documents

# robimy chunki zeby mozna bylo przetworzyc duze dokumenty
# chunk_size - rozmiar chunka, chunk_overlap - ile zapamietuje z poprzedniego chunka, dobre do kontekstu
def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start_chunk = 0
    while start_chunk < len(text):
        end_chunk = start_chunk + chunk_size
        chunk = text[start_chunk:end_chunk]
        chunks.append(chunk)
        start_chunk += chunk_size - chunk_overlap
    return chunks