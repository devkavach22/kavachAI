# llm.py
"""Centralized LLM initialization.

Provides a singleton instance of the Ollama LLM that can be imported
by any module that needs to interact with the model.
"""

import os
import time
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

_ollama_llm_instance = None
_openai_llm_instance = None


def get_llm():
    """Return a cached ChatOllama instance.

    The first call creates the LLM using environment variables for model
    name and API base. Subsequent calls return the same instance, avoiding
    repeated initialisation overhead.
    """
    global _ollama_llm_instance
    if _ollama_llm_instance is None:
        _ollama_llm_instance = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            temperature=0.2,
            max_tokens=50000,
            base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        )
    return _ollama_llm_instance


def get_openai_llm():
    """Return a cached ChatOpenAI instance.

    Similar to get_llm, this provides a singleton for the OpenAI model.
    """
    global _openai_llm_instance
    if _openai_llm_instance is None:
        _openai_llm_instance = ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "gpt-4o"),
            temperature=0.2,
            # api_key is automatically read from OPENAI_API_KEY env var
        )
    return _openai_llm_instance


def main():
    """Run a simple test of the LLM singleton instance.

    This function obtains the cached LLM via ``get_llm`` and prints a brief
    representation to confirm that the instance was created successfully.
    """
    llm = get_llm()
    print("Ollama instance created:", llm)

    openai_llm = get_openai_llm()
    print("OpenAI instance created:", openai_llm)

    print("Invoking LLM...")
    start_time = time.time()
    response = llm.invoke("what is define a user-agent in browser?")
    end_time = time.time()
    print("LLM response:", response.content)
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = duration % 60
    print(
        f"😎 Response time: {minutes} min {seconds:.2f} sec (started at {time.strftime('%X', time.localtime(start_time))}, ended at {time.strftime('%X', time.localtime(end_time))})"
    )


if __name__ == "__main__":
    main()
