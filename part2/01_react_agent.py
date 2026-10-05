import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_config import get_llm
from tools.web_search import search_internet, read_webpage

SYSTEM = (
    "You are a Business Research Agent. Think step by step. "
    "You have exactly two tools: search_internet and read_webpage. "
    "Never call any other tool. "
    "Use search_internet for current data, prices, or news. "
    "Use read_webpage with a URL from the search results if you need more detail. "
    "If a search returns weak results, rephrase the query and search again. "
    "Finish with a short answer that cites the sources you used."
)

llm = get_llm()

try:
    from langchain.agents import create_agent
    agent = create_agent(model=llm, tools=[search_internet, read_webpage], system_prompt=SYSTEM)
except ImportError:
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, [search_internet, read_webpage], prompt=SYSTEM)

question = "How many electric cars were sold in India in September 2026, and which brand led?"
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

# Show the reasoning loop: Action -> Observation -> Answer
for msg in result["messages"]:
    kind = msg.type.upper()
    if getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            print(f"\n[ACTION] {call['name']}({call['args']})")
    elif kind == "TOOL":
        print(f"\n[OBSERVATION] {str(msg.content)[:300]}...")
    elif kind == "AI" and msg.content:
        print(f"\n[ANSWER] {msg.content}")