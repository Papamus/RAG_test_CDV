import streamlit as st
import os
import shutil
from chat_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from docloader import load_pdf, chunk_text
from embeddings import create_index, retrieve_docs

st.set_page_config(layout="wide", page_title="D&D RAG Chatbot App")
st.title("D&D RAG Chatbot app")

with st.popover("Basic chatbot info"):
    st.text("Chatbot made for explaining D&D rules via OpenRouter & LangChain! Also provides a simple story that user can play")
    st.checkbox("Got it!")

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "D&D Assistant"

app_mode = st.radio("Choose Mode:", ["D&D Assistant", "Story Mode"], horizontal=True)

if st.session_state.current_mode != app_mode:
    st.session_state.current_mode = app_mode
    if app_mode == "Story Mode":
        st.session_state["messages"] = [{"role": "assistant", "content": "What character are you playing? Choose the race: Human, Elf, Dwarf, Halfling, Dragonborn, Orc"}]
    else:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you dear Player?"}]
    st.rerun()

api_key = st.secrets["API_KEY"] 
base_url = st.secrets["BASE_URL"]
selected_model = "gemini-3-flash" 

# Inicjalizacja modelu za pomocą klasy ChatOpenRouter
model = ChatOpenRouter(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name=selected_model
)

UPLOAD_FOLDER = "data/uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Szablon asystenta
assistant_template = """
You are a helpful D&D rules expert assistant.
Use the following pieces of context extracted from official rulebooks to answer the user's question.
If you don't know the answer or if it's not in the context, use your general knowledge but mention that it wasn't found in the uploaded documents.

Context: 
{context}
"""

# Szablon Mistrza Gry
story_template = """
You are a Dungeon Master leading a short, interactive D&D adventure for one player. 

### ADVENTURE RULES:
1. CHARACTER CREATION: The user will choose a race. Provide them with the race's characteristics based on your provided PDF context. If it's not in the context, use your base knowledge but kindly inform the player that the source wasn't found in the documents. Guide them to complete their class and basic stats. Do not start the adventure until character creation is fully complete.
2. THE PLOT: Make up a short story involving a robbery and escaping from a location. Randomize the setting entirely (e.g., a medieval castle, a sand village, a localized mansion in another world). Ensure the setting and architecture make narrative sense.
3. THE GOAL: The player's main objective is to obtain the "Golden Turtle". You must creatively invent what this Golden Turtle is, why it is golden, and its significance depending on the specific location and story.
4. GAMEPLAY: The scenario should involve sneaking around, interacting with random NPCs and objects, and potentially fighting. Call for appropriate D&D skill checks (e.g. Stealth, Perception, standard combat) and apply D&D rules correctly.
5. ENDINGS: The adventure ends in success if the player obtains the Golden Turtle and escapes. It ends in failure if the player is defeated or captured. 

Describe the environment, react to the player's choices, and ask them what they want to do next. Use the context below if the player asks for rule clarifications or when verifying traits during character creation.

Context:
{context}
"""



# Funkcja pomocnicza generująca odpowiedź z kontekstem i historią
def answer_question(messages, documents, model, template_text):
    context = ""
    if documents:
        context = "\n\n".join([f"Source: {doc['file_name']}\nContent: {doc['content']}" for doc in documents])
    else:
        context = "Brak wgrywanych dokumentów w bazie wiedzy."
        
    system_prompt = template_text.format(context=context, question="")
    
    # Przekształć słowniki sesji na obiekty langchain
    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
            
    return model.invoke(langchain_messages)

# Stany sesji
if "messages" not in st.session_state:
    if st.session_state.current_mode == "Story Mode":
        st.session_state["messages"] = [{"role": "assistant", "content": "What character are you playing? Choose the race: Human, Elf, Dwarf, Halfling, Dragonborn, Orc"}]
    else:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you dear Player?"}]
if "documents_db" not in st.session_state:
    st.session_state["documents_db"] = []
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None
if "clear_files" not in st.session_state:
    st.session_state.clear_files = False
if "retrieve_files" not in st.session_state:
    st.session_state.retrieve_files = False


with st.expander("Zarządzanie dokumentami (Wgraj zasady D&D)", expanded=False):
    uploaded_files = st.file_uploader("Ładuj PDF(y)", type=["pdf"], accept_multiple_files=True, key="file_uploader")

    if st.button("Usuń pliki"):
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        st.session_state.documents_db = []
        st.session_state.faiss_index = None
        st.session_state.clear_files = True
        st.session_state.retrieve_files = False
        st.success("Pliki i indeks wyczyszczone")
        st.rerun()

if st.session_state.clear_files:
    uploaded_files = None
    st.session_state.clear_files = False

# Przetwarzanie wgranych plików PDF
if uploaded_files:
    new_files_added = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if not any(d.get("file_name", "").startswith(uploaded_file.name) for d in st.session_state["documents_db"]):
            with st.spinner(f"Przetwarzanie {uploaded_file.name}..."):
                try:
                    pdf_text = load_pdf(uploaded_file)
                    text_chunks = chunk_text(pdf_text, chunk_size=1000, chunk_overlap=200)
                    
                    for idx, chunk in enumerate(text_chunks):
                        st.session_state["documents_db"].append({
                            "file_name": f"{uploaded_file.name} (cz. {idx+1})",
                            "content": chunk,
                        })
                    new_files_added = True
                except Exception as e:
                    st.error(f"Błąd pliku {uploaded_file.name}: {e}")

    if new_files_added:
        with st.spinner("Aktualizacja indeksu FAISS..."):
            try:
                st.session_state["faiss_index"] = create_index(st.session_state["documents_db"])
                st.session_state.retrieve_files = True
                st.success("Pliki przeliczone i dodane do FAISS!")
                st.rerun()
            except Exception as e:
                st.error(f"Błąd indeksowania: {e}")

# Wyświetlanie historii czatu
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Obsługa wejścia użytkownika i generowanie odpowiedzi
if question := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    
    active_template = story_template if app_mode == "Story Mode" else assistant_template
    used_files = []

    # Przeszukiwanie bazy i generowanie odpowiedzi
    if st.session_state.retrieve_files and st.session_state["faiss_index"] is not None:
        with st.spinner("Przeszukiwanie bazy wiedzy..."):
            related_documents = retrieve_docs(question, st.session_state["faiss_index"], k=3)
            
            ai_response = answer_question(st.session_state.messages, related_documents, model, active_template)
            answer_content = ai_response.content
            
            # Zapisujemy użyte źródła dokumentów (usuwamy "(cz. index)" by mieć tylko nazwy oryginalnych plików)
            used_files = list(set([doc['file_name'].split(' (cz.')[0] for doc in related_documents]))
    else:
        # Jeśli nie ma dokumentów, generujemy odpowiedź bez kontekstu
        ai_response = answer_question(st.session_state.messages, [], model, active_template)
        answer_content = ai_response.content

    # Dodajemy powiadomienie o źródłach na koniec wiadomości jeśli korzystaliśmy z bazy
    if used_files:
        answer_content += f"\n\n---\n**📚 Użyte źródła RAG:** {', '.join(used_files)}"

    # Zapis i wyświetlenie odpowiedzi asystenta
    st.session_state.messages.append({"role": "assistant", "content": answer_content})
    st.chat_message("assistant").write(answer_content)


# Wyświetlanie załadowanych dokumentów w bazie wiedzy
st.write("---")
with st.expander("Załadowane fragmenty w bazie wiedzy (FAISS):"):
    if st.session_state["documents_db"]:
        for item in st.session_state["documents_db"]:
            st.text(f"🔹 {item['file_name']}: {item['content'][:100]}...")
    else:
        st.info("Baza wiedzy jest pusta.")