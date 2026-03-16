from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.researcher import researcher_node
from agents.critic import critic_node, should_revise
from agents.synthesizer import synthesizer_node


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("researcher", researcher_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("researcher")
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", should_revise, {
        "researcher": "researcher",
        "synthesize": "synthesizer"
    })
    g.add_edge("synthesizer", END)

    return g.compile()


research_graph = build_graph()
