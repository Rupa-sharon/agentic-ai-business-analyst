import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm

llm = get_llm(temperature=0.3)
parser = StrOutputParser()

# Chain 1: Researcher
research_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a market researcher. Reply with 5 concise factual bullet points."),
        ("human", "Topic: {topic}"),
    ])
    | llm | parser
)

# Chain 2: Writer
writer_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a business blog writer. Write a short, clear blog post (about 150 words) from the notes."),
        ("human", "Notes:\n{notes}"),
    ])
    | llm | parser
)

# Chain 3: Proofreader
proof_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a proofreader. Fix grammar and spelling. Return only the corrected text."),
        ("human", "{draft}"),
    ])
    | llm | parser
)

# Output of each chain becomes the input of the next
pipeline = (
    research_chain
    | (lambda notes: {"notes": notes})
    | writer_chain
    | (lambda draft: {"draft": draft})
    | proof_chain
)

print(pipeline.invoke({"topic": "Electric vehicle market in India"}))