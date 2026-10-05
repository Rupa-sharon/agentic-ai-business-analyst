import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm
from tools.web_search import search_internet
from tools.pdf_rag import search_company_pdf


def _make_agent(tools, system):
    llm = get_llm(max_tokens=700)
    try:
        from langchain.agents import create_agent
        return create_agent(model=llm, tools=tools, system_prompt=system)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools, prompt=system)


def _final_text(result):
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return content


def _show_steps(name, result):
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            print(f"   [{name} ACTION] {call['name']}({call['args']})")


def _run(agent, question):
    return agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 12},
    )


# ---------- Agent 1: Research (live web) ----------
RESEARCH_SYSTEM = (
    "You are a Market Research Agent. You have exactly one tool: search_internet. "
    "Never call any other tool. Call search_internet with a plain-text query string. "
    "If results are weak, rephrase and search again (at most 3 searches). "
    "Reply with 5 to 8 concise bullet points of facts, each with its source name and date if known. "
    "Only use facts that appear in the search results."
)


def run_research(question):
    agent = _make_agent([search_internet], RESEARCH_SYSTEM)
    result = _run(agent, question)
    _show_steps("RESEARCH", result)
    return _final_text(result)


# ---------- Agent 2: Analyst (private PDF) ----------
ANALYST_SYSTEM = (
    "You are a Company Analyst Agent. You have exactly one tool: search_company_pdf, "
    "which searches the company's private annual report. Never call any other tool. "
    "Search the PDF 2 to 4 times with different keyword queries (for example: 'electric vehicle sales', "
    "'market share', 'revenue', 'outlook'). Reply with 5 to 8 bullet points of the company's own figures "
    "and strategy, and include the page number [Page N] for every fact. Do not invent figures."
)


def run_analyst(question):
    agent = _make_agent([search_company_pdf], ANALYST_SYSTEM)
    result = _run(agent, question)
    _show_steps("ANALYST", result)
    return _final_text(result)


# ---------- Writer and Reviewer (plain chains, no tools) ----------
_parser = StrOutputParser()

writer_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a business report writer. Write a structured business analysis under 300 words "
         "with these sections: Executive Summary, Market Context (from web research), "
         "Company Position (from the annual report), Opportunities and Risks, Recommendation. "
         "Use ONLY the facts provided. "
         "Keep source labels: [Web] for web research and [Report p.N] for the annual report."),
        ("human",
         "Question: {question}\n\nWEB RESEARCH:\n{research}\n\nANNUAL REPORT FINDINGS:\n{analysis}"),
    ])
    | get_llm(temperature=0.3, max_tokens=800)
    | _parser
)

reviewer_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict reviewer. Check the draft against the source material. "
         "Remove or flag any claim not supported by the sources, and point out where web data and "
         "report data cover different time periods or definitions. "
                  "Return the final report under 250 words, then a section called 'Reviewer Notes' "
         "with at most 3 bullets of one short sentence each."),
        ("human",
         "DRAFT:\n{draft}\n\nWEB RESEARCH:\n{research}\n\nANNUAL REPORT FINDINGS:\n{analysis}"),
    ])
    | get_llm(temperature=0, max_tokens=950)
    | _parser
)