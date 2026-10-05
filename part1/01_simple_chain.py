from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Business Research Analyst. "
     "Only answer business and market questions. "
     "If asked anything else, reply: 'Outside my domain.'"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": "What are the main growth drivers of the EV market in India?"}))
print("---")
print(chain.invoke({"question": "What is a good home remedy for a cold?"}))