import os
import fitz
import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['pl', 'en'], gpu=False)  # Inicjalizacja czytnika OCR dla języka polskiego

def load_pdf(doc_file):
    doc_file.seek(0)
    doc_bytes = doc_file.read()
    doc = fitz.open(stream=doc_bytes, filetype="pdf")
    text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Try to get native text first
        page_text = page.get_text().strip()
        
        # 2. If the page has very little text, it's probably a scanned image
        if len(page_text) < 50:
            print(f"Brak tekstu na stronie {page_num + 1}, uruchamiam OCR...")
            
            # Render the whole page to a high-resolution image (matrix)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better OCR quality
            
            # Convert PyMuPDF pixmap to a numpy array for EasyOCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Run EasyOCR on the image array
            # detail=0 tells it to just return the text strings, not bounding boxes
            ocr_results = reader.readtext(img_np, detail=0) 
            
            # Join the detected lines of text
            page_text = "\n".join(ocr_results)
            
        text += page_text + "\n\n"
        
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