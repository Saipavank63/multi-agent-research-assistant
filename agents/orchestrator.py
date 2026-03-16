from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from agents.researcher import run_researcher
from agents.critic import run_critic
from agents.synthesizer import run_synthesizer


class ResearchState(TypedDict):
    query: str
    messages: Annotated[List, add_messages]
    research_results: List[str]
    critique_score: float
    critique_feedback: str
    final_answer: str
    iteration_count: int


def researcher_node(state: ResearchState) -> dict:
    print(f"[Orchestrator] → Researcher (iteration {state['iteration_count'] + 1})")
    results = run_researcher(state["query"], state.get("critique_feedback", ""))
    return {
        "research_results": results,
        "iteration_count": state["iteration_count"] + 1,
        "messages": [AIMessage(content=f"Researcher found {len(results)} documents")]
    }


def critic_node(state: ResearchState) -> dict:
    print("[Orchestrator] → Critic")
    score, feedback = run_critic(state["query"], state["research_results"])
    return {
        "critique_score": score,
        "critique_feedback": feedback,
        "messages": [AIMessage(content=f"Critic score: {score:.2f}")]
    }


def synthesizer_node(state: ResearchState) -> dict:
    print("[Orchestrator] → Synthesizer")
    answer = run_synthesizer(state["query"], state["research_results"])
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)]
    }


def should_retry(state: ResearchState) -> str:
    MAX_ITERATIONS = 3
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "synthesize"
    if state["critique_score"] >= 0.7:
        return "synthesize"
    return "retry"


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retry": "researcher", "synthesize": "synthesizer"}
    )
    graph.add_edge("synthesizer", END)
    return graph.compile()


def run_research_pipeline(query: str) -> str:
    app = build_graph()
    initial_state: ResearchState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "research_results": [],
        "critique_score": 0.0,
        "critique_feedback": "",
        "final_answer": "",
        "iteration_count": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]


if __name__ == "__main__":
    result = run_research_pipeline("What are the latest advances in agentic AI?")
    print("\n=== FINAL ANSWER ===\n", result)
