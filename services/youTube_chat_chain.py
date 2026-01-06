from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# from llm import get_llm
import os

llm = ChatOpenAI(model_name=os.getenv("OPENROUTER_MODEL"), temperature=0)


def get_youtube_summary_chain():
    """
    Creates a chain to summarize YouTube video transcripts.

    Returns:
        Chain: A LangChain chain that takes 'transcript' as input and returns a short summary.
    """

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that summarizes YouTube videos based on their transcripts.
        
        Please provide a concise and short summary of the following transcript:

        and fullTranscript  will be any language. like hindi, english, gujarati, spanish, french, german, chinese, japanese, arabic, russian, portuguese, italian, korean, dutch
        
        If the fullTranscript is not in English, then summarize the English version.
        

        {fullTranscript}
        
        Summary:
        """
    )

    chain = prompt | llm | StrOutputParser()

    return chain


def summarize_transcript(fullTranscript: str) -> str:
    """
    Summarizes the given transcript using the LLM chain.


    Args:
        transcript (str): The text content of the YouTube transcript.

    Returns:
        str: The generated summary.
    """
    chain = get_youtube_summary_chain()
    return chain.invoke({"fullTranscript": fullTranscript})


def get_youtube_qa_chain():
    """
    Creates a chain to answer questions about YouTube video transcripts.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """
You are an expert Question-Answering Assistant specializing in analyzing YouTube video transcripts. 
Your goal is to be helpful and informative by always providing the most relevant and complete information to the user's question, drawing from both the transcript and external knowledge.

**Core Rules for Answering:**
1.  **Grounded Extraction:** Use the provided `Video Transcript` as the **primary context** for the topic, summarizing or quoting where appropriate.
2.  **Completeness over refusal (Internal Data):** If the transcript does *not* contain the specific answer, you **must not** refuse to answer. Instead, provide a highly relevant **summary or the closest key points** from the transcript that are thematically related to the user's question.
3.  **Mandatory Knowledge Integration:** You **must** use your general, external knowledge to **supplement, enrich, or clarify** the information found in the transcript, even when the transcript provides a direct answer.
4.  **Conciseness:** Keep your answer clear, direct, and limited to **2 to 4 sentences**.
5.  **Language:** Always respond in **English**. If the source transcript is not English, translate the relevant parts to English before formulating your final answer.
"""
            ),
            (
                "user",
                """
### Video Transcript:
{transcript}

### User Question:
{question}
"""
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain



def answer_transcript(question: str, transcript: str) -> str:
    """
    Answers the given question using the LLM chain.

    Args:
        question (str): The user's question.
        transcript (str): The text content of the YouTube transcript.

    Returns:
        str: The generated answer.
    """
    chain = get_youtube_qa_chain()
    return chain.invoke({"question": question, "transcript": transcript})
