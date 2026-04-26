import streamlit as st
import pandas as pd
from io import StringIO
from openai import OpenAI
from docloader import load_pdf, load_docs_from_folder
import os

st.set_page_config(layout="wide", page_title="Gemini chatbot app")
st.title("Gemini chatbot app")

with st.popover("Basic chatbot info"):
    st.text("Chatbot made for explaining D&D rules!")
    st.checkbox("Got it!")
# api_key, base_url = os.environ["API_KEY"], os.environ["BASE_URL"]
api_key, base_url = st.secrets["API_KEY"], st.secrets["BASE_URL"]
selected_model = "gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model = selected_model,
        messages = st.session_state.messages
    )

    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


with st.expander("Show file uploader"):
    file_uploader = st.file_uploader(
        "Upload images or Excel files", accept_multiple_files = True, type=["xlsx", "jpg", "png", "pdf"]
    )

    if file_uploader:
        for uploaded_files in file_uploader:
            if uploaded_files.name.lower().endswith(".pdf"):
                
                # Unikamy duplikowania tych samych plików w sesji
                if not any(msg.get("file_name") == uploaded_file.name for msg in st.session_state.messages):
                    
                    with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                        try:
                            # Wywołanie funkcji z drugiego pliku
                            pdf_text = load_pdf(uploaded_file)
                            
                            # Dodanie do historii sesji
                            st.session_state.messages.append({
                                "role": "system",
                                "content": f"Context from {uploaded_file.name}: {pdf_text}",
                                "file_name": uploaded_file.name
                            })
                            st.success(f"Dodano: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Błąd pliku {uploaded_file.name}: {e}")
            
            elif uploaded_file.name.lower().endswith((".jpg", ".png", ".xlsx")):
                st.info(f"Plik {uploaded_file.name} wykryty, ale obsługa PDF jest priorytetem.")

# Podgląd tego, co siedzi w sesji
if st.session_state.messages:
    st.write("Zawartość sesji (Skrót):")
    for item in st.session_state.messages:
        if "file_name" in item:
            st.text(f"{item['file_name']}: {item['content'][:100]}...")

