from langchain_community.embeddings import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

vectors = embeddings.embed_documents(
    ["This is the first text.", "Another text for embedding.", "LangChain is great!"]
)

print("Embedded Vectors:")
print(
    len(vectors), len(vectors[0]), len(vectors[1]), len(vectors[2])
)  # e.g. 3 embeddings of dimension 1536
# print(vectors)  # Print the actual vectors

print("vector data type:", type(vectors[0][0]))
print("vector data type:", type(vectors))
