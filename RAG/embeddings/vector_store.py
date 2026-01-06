from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

docs = """If you provide embeddings, ChromaDB will use those directly.
If you do not provide embeddings, ChromaDB will automatically generate them using the embedding function configured for the collection. By default, it uses a sentence transformer model like all-MiniLM-L6-v2.
Ensure that the lists for documents, metadatas, ids, and embeddings (if provided) have the same length and correspond to each other."""

chroma_db = Chroma(
    persist_directory="data",
    embedding_function=embeddings,
    collection_name="chat_docs_collection",
)

chroma_db.add_texts([docs], metadatas=[{"source": "sample_doc_1"}])
chroma_db.persist()

addedData = chroma_db.from_texts(
    [docs], embedding=embeddings, collection_name="chat_docs_collection"
)
print("Added Data: -->>> ", addedData.get())

collection = chroma_db.get()
print("Collection info:", collection)
