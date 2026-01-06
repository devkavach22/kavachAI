import chromadb
# from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="./chroma_db")

model = SentenceTransformer("all-MiniLM-L6-v2")

def local_embedding_function(texts):
    return model.encode(texts).tolist()

def create_collection_if_not_exists(collection_name: str):
    if collection_name in [col.name for col in chroma_client.list_collections()]:
        return chroma_client.get_collection(name=collection_name)
    else:
        return chroma_client.create_collection(name=collection_name)


CV_data_collection = create_collection_if_not_exists("CV_data_collection")

if __name__ == "__main__":
    print("Collections:", chroma_client.list_collections())
    
    # print("Documents data.", CV_data_collection.get(include=["embeddings", "documents", "metadatas"])) 
    # CV_data_collection.delete(ids=["id_7102458025"])
    print("Documents data after deletion.", CV_data_collection.get())
    # results = CV_data_collection.query(
    #         query_texts=[query],
    #         n_results=2,
    #         where={"chat_ID": chat_ID}  # Filter by metadata
    #     )

    # Check if chat_ID exists in the collection
    chat_ID_check = "some_chat_id_to_check"
    existing_docs = CV_data_collection.get(where={"chat_ID": chat_ID_check})
    
    if existing_docs['ids']:
        print(f"Chat ID {chat_ID_check} is present.")
    else:
        print(f"Chat ID {chat_ID_check} is not found.")



