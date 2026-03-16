import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools.vector_search import hybrid_search, rerank

llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

SYSTEM = """You are a research agent. Given a query and retrieved document chunks,
extract and summarize the most relevant facts. Be precise and cite the source when possible.
Do not fabricate information outside the provided context."""


def researcher_node(state: dict) -> dict:
    query = state["query"]

    docs = hybrid_search(query)
    docs = rerank(docs)

    context = "\n\n".join(f"[Source: {d['source']}]\n{d['text']}" for d in docs)

    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Query: {query}\n\nContext:\n{context}")
    ])

    return {
        "retrieved_docs": docs,
        "research_notes": response.content,
        "messages": [{"role": "researcher", "content": response.content}]
    }
