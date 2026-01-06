from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from llm.base import get_llm
import os

def get_home_page_chain():
    """
    Creates and returns a LangChain chain for the home page.
    The chain is designed to handle user queries, file data, or both.
    """
    # llm = get_llm()
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0.2)

    template = """You are a brilliant AI assistant. You help users understand their data and answer their questions with precision and insight.

    Context Information:
    {file_data}

    User Query:
    {user_query}

    Instructions:
    1. If file data is provided but no user query, provide a comprehensive yet concise overview of the file's content.
    2. If both file data and a user query are provided, answer the query specifically using the file data as context.
    3. If only a user query is provided, answer it to the best of your knowledge.
    4. Always maintain a professional, helpful, and "brilliant" tone.
    5. If the file data seems to be code, explain its purpose and key components.
    6. If the file data is a table (CSV/Excel), summarize the key trends or data points.

    Response:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    return chain


async def run_home_page_chain(
    user_query: Optional[str] = None, file_data: Optional[str] = None
):
    """
    Runs the home page chain with the provided inputs.
    """
    chain = get_home_page_chain()

    # Handle empty cases
    query = user_query if user_query else "No specific query provided."
    data = file_data if file_data else "No file data provided."

    # Run the chain
    # Note: Using invoke for now, can be swapped for astream in websocket integration
    response = await chain.ainvoke({"user_query": query, "file_data": data})

    return response
