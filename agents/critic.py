from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import ResearchState
from config import LLM_MODEL, OPENAI_API_KEY

llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)

SYSTEM = """You are a critical reviewer. Your job is to evaluate research notes for:
- Factual accuracy based on the provided context
- Missing key information
- Unsupported claims

Respond with: PASS if the notes are solid, or REVISE: <reason> if they need improvement."""


def critic_node(state: ResearchState) -> dict:
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


def should_revise(state: ResearchState) -> str:
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)

    if iteration >= 2:
        return "synthesize"
    if critique.strip().startswith("REVISE"):
        return "researcher"
    return "synthesize"
