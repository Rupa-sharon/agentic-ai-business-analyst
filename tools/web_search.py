import os
import re
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def search_internet(query: str) -> str:
    """Look up current information on the internet using Google (via Serper).
    Input must be a plain-text search query string, for example 'India EV sales September 2026'.
    Use this for current news, prices, market data, or competitor information."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        },
        json={"q": query, "num": 5},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("organic", [])

    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})")
    return "\n".join(lines)


@tool
def read_webpage(url: str) -> str:
    """Open a web page and return its readable text.
    Input must be a full URL starting with http, taken from search_internet results.
    Use this when search snippets are not detailed enough."""
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"Could not open the page: {e}"
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", r.text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]