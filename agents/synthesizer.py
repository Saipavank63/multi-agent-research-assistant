import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    streaming=True
)

SYSTEM = """You are a synthesis agent. Combine the research notes into a clear, well-structured
answer for the original query. Use markdown formatting. Cite sources inline as [Source: filename]."""


def synthesizer_node(state: dict) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Query: {state['query']}\n\n"
            f"Research notes:\n{state['research_notes']}\n\n"
            f"Critic feedback:\n{state.get('critique', 'N/A')}"
        ))
    ])

    return {
        "final_answer": response.content,
        "messages": [{"role": "synthesizer", "content": response.content}]
    }
