# app/services/qa_response.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
    temperature=0.2,
    max_tokens=8000,
    openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
    openai_api_key=os.getenv("OLLAMA_API_KEY")
)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that answers user questions based on the provided query.",
#     ),
#     ("human", "what is a python"),
# ]
# ai_msg = llm.invoke(messages)
# print("ai_msg -->>",ai_msg.content)

def run_qa_query(query: str, context, detail: bool = True) -> str:
    """
    Runs a QA query with optional 'detail mode'.
    detail = True   -> long, deep, structured, ChatGPT-style answers
    detail = False  -> short, concise answers
    """

    # Normalize context
    if isinstance(context, list):
        context = "\n\n".join([str(c).strip() for c in context if c])
    elif not isinstance(context, str):
        context = str(context)


    # Detail mode
    should_be_concise = len(query) <= 100 or '?' in query
    detail_instruction = (
        "Provide a short and concise answer."
        if should_be_concise or not detail
        else "Provide a detailed and well-structured answer."
    )


    prompt_template = """
You are an intelligent AI assistant designed to analyze PDF documents and answer user questions with clarity, depth, and precision.

Your Goal: Provide a natural, helpful, and "ChatGPT-like" response.

Instructions:
- **Be Adaptive**: If the user asks a simple question, give a concise answer. If they ask for an explanation, provide a detailed, well-structured response.
- **Use Markdown**: Use **bold** for emphasis, lists (bullet points/numbered) for readability, and headers (##) to organize long answers.
- **Grounding**: Answer strictly based on the provided "PDF Context". If the answer is not in the context, politely state that the information is not available in the document.
- **Tone**: Professional, friendly, and direct. Avoid robotic or repetitive phrases.
- **Simplicity**: Explain concepts in plain, layman's terms. Avoid jargon where possible, or explain it if necessary.
- **No Forced Structure**: Do NOT force every answer into an "Executive Summary" or "Key Insights" format unless it makes sense for the specific query.

Extra Instruction: {detail_instruction}

===================================
### 📄 PDF Context:
{context}

### ❓ User Query:
{question}
===================================
"""


    prompt = PromptTemplate(
        input_variables=["context", "question","detail_instruction"],
        template=prompt_template
    )

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1"),   # UPGRADED!
        temperature=0.2,
        max_tokens=8000,
        openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
    )

    final_prompt = prompt.format(context=context, question=query,detail_instruction=detail_instruction)
    response = llm.invoke(final_prompt)

    return response.content.strip()

