from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research synthesizer. Your job is to produce a clear,
accurate, well-structured answer based ONLY on the provided documents.

Instructions:
1. Identify the key claim or insight from each relevant document
2. Resolve any contradictions between sources explicitly
3. Structure the answer with a direct response first, then supporting detail
4. Flag any important gaps or uncertainties at the end
5. Do not invent information not present in the documents"""),
    ("human", """Query: {query}

Research Documents:
{documents}

Provide a comprehensive, well-structured answer:""")
])


def run_synthesizer(query: str, research_results: List[str]) -> str:
    if not research_results:
        return "Insufficient information found to answer this query reliably."

    docs_text = "\n\n---\n\n".join(
        f"Source {i+1}:\n{doc[:800]}"
        for i, doc in enumerate(research_results[:8])
    )

    try:
        response = llm.invoke(
            SYNTHESIZER_PROMPT.format_messages(query=query, documents=docs_text)
        )
        return response.content.strip()
    except Exception as e:
        print(f"[Synthesizer] Error: {e}")
        return f"Synthesis failed: {str(e)}"
