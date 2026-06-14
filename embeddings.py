import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = "intfloat/e5-small-v2" # nazwa modelu
model_kwargs = {"device": "cpu", "trust_remote_code": True}

# Funkcja z dekoratorem - ładuje model tylko raz, a nie za kadym razem gdy user poda jakieś query
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

def create_index(documents):
    embeddings = get_embedding_model() # załadowanie modelu embeddingowego
    texts = [doc['content'] for doc in documents] # wartości tekstowe wszystkich dokumentów
    metadata = documents
    embeddings_matrix = [embeddings.embed_query(f"passage: {text}") for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    matrix_dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(matrix_dim)# ustawienie indeksu przeszukwania
    index.add(embeddings_matrix)

    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = get_embedding_model() # załadowanie modelu embeddingowego

    formatted_query = f"query: {query}"
    query_embedding = np.array([embeddings.embed_query(formatted_query)]).astype("float32") # embeddowanie zapytania (query)
    results = faiss_index.similarity_search(query_embedding, k=k) # zwrócenie wyników przeuszkiwania
    return results