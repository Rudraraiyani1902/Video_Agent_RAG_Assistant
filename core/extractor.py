# Actionable items, decision, questions & unified analysis
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    primary_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    fallback_model = "gemini-3.5-flash" if "lite" in primary_model else "gemini-3.5-flash-lite"

    primary_llm = ChatGoogleGenerativeAI(
        model=primary_model,
        google_api_key=api_key,
        temperature=0.2,
    )
    fallback_llm = ChatGoogleGenerativeAI(
        model=fallback_model,
        google_api_key=api_key,
        temperature=0.2,
    )
    return primary_llm.with_fallbacks([fallback_llm])


def analyze_transcript(transcript: str) -> dict:
    """
    Unified analysis: extracts title, summary, action items, decisions,
    and open questions in a SINGLE API call, saving ~80% of quota and
    running 5x faster.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert executive meeting and video content analyst.
Analyze the provided transcript and extract all insights in one comprehensive, structured pass.
Return ONLY valid raw JSON with the following exact keys:
"title": Short professional title for the session (max 8 words)
"summary": Structured summary using clear markdown bullet points covering key discussions and insights
"action_items": Numbered list of action items with task description, owner, and deadline (or 'No action items found.')
"key_decisions": Numbered list of all decisions made (or 'No key decisions found.')
"open_questions": Numbered list of unresolved questions or topics needing follow-up (or 'No open questions found.')

Do not include any text outside the JSON. Do not wrap in markdown code blocks."""),
        ("human", "{text}")
    ])

    chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | prompt | llm | StrOutputParser()
    raw_output = chain.invoke(transcript)

    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return {
            "title": data.get("title", "Meeting Analysis"),
            "summary": data.get("summary", ""),
            "action_items": data.get("action_items", "No action items found."),
            "key_decisions": data.get("key_decisions", "No key decisions found."),
            "open_questions": data.get("open_questions", "No open questions found."),
        }
    except Exception:
        return {
            "title": "Meeting Analysis",
            "summary": extract_action_items(transcript),
            "action_items": extract_action_items(transcript),
            "key_decisions": extract_key_decisions(transcript),
            "open_questions": extract_questions(transcript),
        }


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)