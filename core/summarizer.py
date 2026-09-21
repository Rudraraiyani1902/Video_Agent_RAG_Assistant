from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 

def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    primary_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    fallback_model = "gemini-3.5-flash" if "lite" in primary_model else "gemini-3.5-flash-lite"

    primary_llm = ChatGoogleGenerativeAI(
        model=primary_model,
        google_api_key=api_key,
        temperature=0.3,
    )
    fallback_llm = ChatGoogleGenerativeAI(
        model=fallback_model,
        google_api_key=api_key,
        temperature=0.3,
    )
    return primary_llm.with_fallbacks([fallback_llm])


def summarize(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting and video content analyst. "
            "Generate a clear, professional, comprehensive summary of the transcript in structured bullet points. "
            "Highlight the main topics, key insights, and important takeaways.",
        ),
        ("human", "{text}"),
    ])

    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )

    # For transcripts up to 200k chars (~40k words), single-pass is 10x faster and uses only 1 API call
    if len(transcript) <= 200000:
        return chain.invoke(transcript)

    # Fallback for extremely long files (>4 hours)
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = splitter.split_text(transcript)
    chunk_summaries = [chain.invoke(chunk) for chunk in chunks]
    return "\n\n".join(chunk_summaries)

def generate_title(transcipt : str) -> str:
    llm = get_llm()

    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])