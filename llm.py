# llm.py
"""Centralized LLM initialization.

Provides a singleton instance of the Ollama LLM that can be imported
by any module that needs to interact with the model.
"""

import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()
import time

_llm_instance = None


def get_llm():
    """Return a cached ChatOllama instance.

    The first call creates the LLM using environment variables for model
    name and API base. Subsequent calls return the same instance, avoiding
    repeated initialisation overhead.
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            temperature=0.2,
            max_tokens=50000,
            base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        )
    return _llm_instance


def main():
    """Run a simple test of the LLM singleton instance.

    This function obtains the cached LLM via ``get_llm`` and prints a brief
    representation to confirm that the instance was created successfully.
    """
    llm = get_llm()
    print("LLM instance created:", llm)

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
