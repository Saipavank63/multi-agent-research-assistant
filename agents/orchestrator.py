import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from agents.researcher import researcher_node
from agents.critic import critic_node
from agents.synthesizer import synthesizer_node


class ResearchState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    research_notes: str
    critique: str
    final_answer: str
    iteration: int
    messages: Annotated[list, operator.add]


def _should_revise(state: ResearchState) -> str:
    if state.get("iteration", 0) >= 2:
        return "synthesize"
    if state.get("critique", "").strip().startswith("REVISE"):
        return "researcher"
    return "synthesize"


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("researcher", researcher_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("researcher")
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", _should_revise, {
        "researcher": "researcher",
        "synthesize": "synthesizer"
    })
    g.add_edge("synthesizer", END)

    return g.compile()


research_graph = build_graph()
