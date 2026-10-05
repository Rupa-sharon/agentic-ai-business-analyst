import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature=0, max_tokens=800):
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
            max_retries=8,
            max_tokens=max_tokens,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=temperature,
        max_tokens=max_tokens,
    )