# app/services/qa_response_ollama.py

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama   # 🔥 Ollama LangChain wrapper

def run_qa_query(query: str, context, detail: bool = True) -> str:
    """
    Runs a QA query using a local OLLAMA LLM.
    detail = True   -> long, deep, structured answers
    detail = False  -> short, concise answers
    """

    # Normalize context
    if isinstance(context, list):
        context = "\n\n".join([str(c).strip() for c in context if c])
    elif not isinstance(context, str):
        context = str(context)

    # Main prompt
    prompt_template = """
You are an intelligent AI assistant designed to analyze PDF documents
and answer questions with clarity and precision.

Your Goal: Provide natural, helpful, and "ChatGPT-like" responses.

Instructions:
- Be adaptive: concise for simple questions, detailed for complex ones.
- Use Markdown for formatting.
- Ground answers only in the given PDF context.
- If answer is missing in the PDF, clearly say so.
- Keep tone professional and human-like.

===================================
### 📄 PDF Context:
{context}

### ❓ User Query:
{question}
===================================
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Select ollama model
    model_name = "gpt-oss:latest"      # Or "mistral", "phi3", "deepseek-r1", etc.

    llm = ChatOllama(
        model=model_name,
        temperature=0.2 if detail else 0.0,
        num_predict=8000
    )

    final_prompt = prompt.format(context=context, question=query)

    response = llm.invoke(final_prompt)

    return response.content.strip()


def main():
    # Example usage for testing the generate_ollama_response function
    example_context = "The small red squirrel gathered acorns near the old oak tree. Later, it climbed to the top branch to look for predators."
    example_query = "What was the squirrel doing near the oak tree?"
    
    # Set detail to True for a more verbose response, False for concise
    detail_level = True 

    print("Generating response from Ollama...")
    response_content = run_qa_query(example_query, example_context, detail=detail_level)
    print("\n--- Ollama Response ---")
    print(response_content)

if __name__ == "__main__":
    main()
