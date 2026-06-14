# import streamlit as st
# import pandas as pd
# from io import StringIO
# from openai import OpenAI
# from docloader import load_pdf, load_docs_from_folder, chunk_text
# from embeddings import create_index, retrieve_docs
# import os

# st.set_page_config(layout="wide", page_title="Gemini chatbot app")
# st.title("Gemini chatbot app")

# with st.popover("Basic chatbot info"):
#     st.text("Chatbot made for explaining D&D rules!")
#     st.checkbox("Got it!")
# # api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
# api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
# selected_model = "gemini-2.5-flash"

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
# if "documents_db" not in st.session_state:
#     st.session_state["documents_db"] = []  # Tu zbieramy czyste słowniki z chunkami
# if "faiss_index" not in st.session_state:
#     st.session_state["faiss_index"] = None  # Tu ląduje obiekt indeksu FAISS

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input():
#     if not api_key:
#         st.info("Invalid API key.")
#         st.stop()
#     client = OpenAI(
#         api_key = api_key,
#         base_url = base_url
#     )
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     response = client.chat.completions.create(
#         model = selected_model,
#         messages = st.session_state.messages
#     )

#     msg = response.choices[0].message.content
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)


# with st.expander("Show file uploader"):
#     file_uploader = st.file_uploader(
#         "Upload images or Excel files", accept_multiple_files = True, type=["xlsx", "jpg", "png", "pdf"]
#     )

#     if file_uploader:
#         new_files_added = False
#         for uploaded_file in file_uploader:
#             if uploaded_file.name.lower().endswith(".pdf"):
                
#                 # Unikamy duplikowania tych samych plików w sesji
#                 if not any(d.get("file_name", "").startswith(uploaded_file.name) for d in st.session_state["documents_db"]):

#                     with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
#                         try:
#                             # Wywołanie funkcji z drugiego pliku
#                             pdf_text = load_pdf(uploaded_file)
#                             text_chunks = chunk_text(pdf_text, chunk_size=500, chunk_overlap=100)
                            
#                             # Dodanie do historii sesji

#                             for idx, chunk in enumerate(text_chunks):
#                                 st.session_state["documents_db"].append({
#                                     "role": "system",
#                                     "file_name": f"{uploaded_file.name} (cz. {idx+1})",
#                                     "content": f"Context from {uploaded_file.name}: {chunk}",
#                                 })
#                             new_files_added = True
#                             st.success(f"Dodano: {uploaded_file.name}")
#                         except Exception as e:
#                             st.error(f"Błąd pliku {uploaded_file.name}: {e}")
            
#             elif uploaded_file.name.lower().endswith((".jpg", ".png", ".xlsx")):
#                 st.info(f"Plik {uploaded_file.name} wykryty, ale obsługa PDF jest priorytetem.")

#         if new_files_added:
#             with st.spinner("Generowanie embeddingów i aktualizacja bazy wektorowej..."):
#                 try:
#                     st.session_state["faiss_index"] = create_index(st.session_state["documents_db"])
#                     st.success("Baza wiedzy została pomyślnie zaembedowana i zindeksowana")
#                 except Exception as e:
#                     st.error(f"Błąd podczas tworzenia indeksu wektorowego: {e}")

# # Podgląd tego, co siedzi w sesji
#     if st.session_state["documents_db"]:
#         for item in st.session_state["documents_db"]:
#             # Wyświetlamy nazwę chunka i jego początek
#             st.text(f"{item['file_name']}: {item['content'][:100]}...")
#     else:
#         st.info("Baza wiedzy jest pusta. Wgraj pliki PDF, aby wygenerować wektory.")
#     if st.session_state.messages:
#         st.write("Zawartość sesji (Skrót):")
#         for item in st.session_state.messages:
#             if "file_name" in item:
#                 st.text(f"{item['file_name']}: {item['content'][:100]}...")
   

  #KOD OD MAKSA DOSTOSOWANY PODE MNIE - do testowania, powyzej moje rozwiazanie, tam do dodania rozne rzeczy z kodu ponizej 

import streamlit as st
import os
import shutil
from chat_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate

# Importy z Twoich modułów pomocniczych
from docloader import load_pdf, chunk_text
from embeddings import create_index, retrieve_docs

st.set_page_config(layout="wide", page_title="Gemini RAG Chatbot")
st.title("Gemini RAG Chatbot app")

# --- POP-OVER INFO ---
with st.popover("Basic chatbot info"):
    st.text("Chatbot made for explaining D&D rules via OpenRouter & LangChain!")
    st.checkbox("Got it!")

# --- KONFIGURACJA SEKRETÓW I MODELU ---
# Pobieramy dane z st.secrets (zgodnie z Twoim oryginalnym kodem)
api_key = st.secrets["API_KEY"] 
base_url = st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"  # Pełna nazwa modelu dla OpenRouter

# Inicjalizacja modelu za pomocą klasy ChatOpenRouter
model = ChatOpenRouter(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name=selected_model
)

# --- FOLDER NA PLIKI ---
UPLOAD_FOLDER = "data/uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- SZABLON PROMPTU (LANGCHAIN) ---
template = """
You are a helpful D&D rules expert assistant.
Use the following pieces of context extracted from official rulebooks to answer the user's question.
If you don't know the answer or if it's not in the context, use your general knowledge but mention that it wasn't found in the uploaded documents.

Context: 
{context}

Question: 
{question}

Answer:
"""

# Funkcja pomocnicza generująca odpowiedź z kontekstem RAG przy użyciu LangChain LCEL
def answer_question(question, documents, model):
    context = "\n\n".join([f"Source: {doc['file_name']}\nContent: {doc['content']}" for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# --- INICJALIZACJA STANÓW SESJI ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "documents_db" not in st.session_state:
    st.session_state["documents_db"] = []
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "clear_files" not in st.session_state:
    st.session_state.clear_files = False
if "retrieve_files" not in st.session_state:
    st.session_state.retrieve_files = False

# --- PASEK BOCZNY: ZARZĄDZANIE PLIKAMI ---
st.sidebar.header("Zarządzanie dokumentami")
uploaded_files = st.sidebar.file_uploader("Ładuj PDF(y)", type=["pdf"], accept_multiple_files=True, key="file_uploader")

if st.sidebar.button("Usuń pliki"):
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    st.session_state.documents_db = []
    st.session_state.faiss_index = None
    st.session_state.clear_files = True
    st.session_state.retrieve_files = False
    st.sidebar.success("Pliki i indeks wyczyszczone")
    st.rerun()

if st.session_state.clear_files:
    uploaded_files = None
    st.session_state.clear_files = False

# --- PRZETWARZANIE WGRANYCH PLIKÓW ---
if uploaded_files:
    new_files_added = False
    for uploaded_file in uploaded_files:
        # 1. Zapisujemy fizycznie plik na dysku serwera (zgodnie z nowym wzorcem)
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # 2. Unikamy duplikacji w naszej bazie obiektów w pamięci podręcznej RAM
        if not any(d.get("file_name", "").startswith(uploaded_file.name) for d in st.session_state["documents_db"]):
            with st.sidebar.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                try:
                    pdf_text = load_pdf(uploaded_file)
                    text_chunks = chunk_text(pdf_text, chunk_size=500, chunk_overlap=100)
                    
                    for idx, chunk in enumerate(text_chunks):
                        st.session_state["documents_db"].append({
                            "file_name": f"{uploaded_file.name} (cz. {idx+1})",
                            "content": chunk,
                        })
                    new_files_added = True
                except Exception as e:
                    st.sidebar.error(f"Błąd pliku {uploaded_file.name}: {e}")

    # 3. Jeśli doszły nowe pliki, przeliczamy indeks FAISS
    if new_files_added:
        with st.sidebar.spinner("Aktualizacja indeksu FAISS..."):
            try:
                st.session_state["faiss_index"] = create_index(st.session_state["documents_db"])
                st.session_state.retrieve_files = True
                st.sidebar.success("Pliki przeliczone i dodane do FAISS!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Błąd indeksowania: {e}")

# --- HISTORIA CZATU (WIDOK) ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- OBSŁUGA CZATU ---
if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    # Przeszukiwanie bazy i generowanie odpowiedzi
    if st.session_state.retrieve_files and st.session_state["faiss_index"] is not None:
        with st.spinner("Przeszukiwanie bazy wiedzy..."):
            # Pobieramy pasujące dokumenty przy użyciu Twojej funkcji
            related_documents = retrieve_docs(question, st.session_state["faiss_index"], k=3)
            
            # Generujemy odpowiedź przez łańcuch LangChain z kontekstem RAG
            ai_response = answer_question(question, related_documents, model)
            answer_content = ai_response.content
    else:
        # Zwykłe wywołanie modelu, gdy baza dokumentów jest pusta
        ai_response = model.invoke(st.session_state.messages)
        answer_content = ai_response.content

    # Zapis i wyświetlenie odpowiedzi asystenta
    st.session_state.messages.append({"role": "assistant", "content": answer_content})
    st.chat_message("assistant").write(answer_content)


# --- SEKCJA PODGLĄDU NA SAMYM DOLE ---
st.write("---")
st.subheader("Załadowane fragmenty w bazie wiedzy (FAISS):")
if st.session_state["documents_db"]:
    for item in st.session_state["documents_db"]:
        st.text(f"🔹 {item['file_name']}: {item['content'][:100]}...")
else:
    st.info("Baza wiedzy jest pusta.")