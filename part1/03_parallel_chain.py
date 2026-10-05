import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from llm_config import get_llm

llm = get_llm(temperature=0.3)
parser = StrOutputParser()

swot_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a strategy analyst. Give a brief SWOT analysis (2 bullets per section)."),
        ("human", "Topic: {topic}"),
    ])
    | llm | parser
)

risk_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a risk analyst. List the top 3 business risks with one line each."),
        ("human", "Topic: {topic}"),
    ])
    | llm | parser
)

# Both chains run at the same time on the same input
parallel = RunnableParallel(swot=swot_chain, risks=risk_chain)

# Combine the two results into one final report
combine_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a business report writer. Merge the SWOT and risks into one short executive summary."),
        ("human", "SWOT:\n{swot}\n\nRisks:\n{risks}"),
    ])
    | llm | parser
)

full_chain = parallel | combine_chain

result = parallel.invoke({"topic": "Electric vehicle market in India"})
print("=== SWOT ===\n", result["swot"])
print("\n=== RISKS ===\n", result["risks"])

print("\n=== EXECUTIVE SUMMARY ===")
print(full_chain.invoke({"topic": "Electric vehicle market in India"}))