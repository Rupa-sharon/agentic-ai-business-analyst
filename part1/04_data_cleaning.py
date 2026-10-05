import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from llm_config import get_llm

llm = get_llm()

def clean_text(text: str) -> str:
    text = text.lower()                       # lowercase
    text = re.sub(r"[^a-z0-9\s.,!?]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

cleaner = RunnableLambda(clean_text)

sentiment_chain = (
    cleaner
    | (lambda cleaned: {"feedback": cleaned})
    | ChatPromptTemplate.from_messages([
        ("system", "Classify customer feedback as Positive, Negative or Neutral and give a one-line reason."),
        ("human", "{feedback}"),
    ])
    | llm
    | StrOutputParser()
)

raw = "   LOVED the   product!!! ###  Delivery was LATE tho...   @@@ "
print("Cleaned text:", clean_text(raw))
print("Model result:", sentiment_chain.invoke(raw))