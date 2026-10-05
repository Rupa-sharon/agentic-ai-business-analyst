import os, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm
from agents.team import run_research, run_analyst, writer_chain, reviewer_chain

DEFAULT_Q = "How is Tata Motors Passenger Vehicles positioned in India's electric vehicle market?"


def pause(seconds=60):
    print(f"   (waiting {seconds}s to respect rate limits...)")
    time.sleep(seconds)


# Conversation memory: the last 2 question/answer pairs
history = []

def history_text():
    return "\n\n".join(
        f"Q: {h['q']}\nA: {h['a'][:600]}" for h in history[-2:]
    )

condense_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's follow-up as ONE standalone business question, using the "
         "conversation history for context. Return only the question."),
        ("human", "History:\n{history}\n\nFollow-up: {question}"),
    ])
    | get_llm(max_tokens=100)
    | StrOutputParser()
)


def run_pipeline(question):
    print("\n[1/4] Research Agent searching the web...")
    research = run_research(
        f"{question} (Focus on Tata Motors Passenger Vehicles and the Indian market.)"
    )
    pause()
    print("\n[2/4] Analyst Agent reading the annual report...")
    analysis = run_analyst(question)
    pause()
    print("\n[3/4] Writer drafting the report...")
    draft = writer_chain.invoke({"question": question, "research": research, "analysis": analysis})
    pause()
    print("\n[4/4] Reviewer checking the draft...")
    final = reviewer_chain.invoke({"draft": draft, "research": research, "analysis": analysis})
    return final, research, analysis


turn = 0
print("Type a business question. Press Enter for the default. Type 'quit' to stop.")
while True:
    question = input("\nYour question: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        question = DEFAULT_Q

    try:
        if history:
            standalone = condense_chain.invoke({"history": history_text(), "question": question}).strip()
            print(f"   (understood as: {standalone})")
            pause(30)
        else:
            standalone = question

        final, research, analysis = run_pipeline(standalone)
    except Exception as e:
        print("\nSTOPPED:", str(e)[:600])
        continue

    history.append({"q": standalone, "a": final})
    turn += 1
    os.makedirs("reports", exist_ok=True)
    path = f"reports/analysis_{turn}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {standalone}\n\n{final}\n\n---\n## Raw web research\n{research}\n\n## Raw report findings\n{analysis}\n")

    print("\n" + "=" * 60)
    print(final)
    print("=" * 60)
    print(f"Saved to {path}")

print("Goodbye.")