import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

SYSTEM = """You are a critical reviewer. Evaluate the research notes for:
- Factual accuracy based on the provided context
- Missing key information
- Unsupported claims

Respond with PASS if the notes are solid, or REVISE: <reason> if they need improvement."""


def critic_node(state: dict) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Original query: {state['query']}\n\n"
            f"Research notes:\n{state['research_notes']}"
        ))
    ])

    return {
        "critique": response.content,
        "messages": [{"role": "critic", "content": response.content}],
        "iteration": state.get("iteration", 0) + 1
    }
