from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Tuple
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)


class CritiqueOutput(BaseModel):
    score: float = Field(description="Quality score 0.0-1.0", ge=0.0, le=1.0)
    feedback: str = Field(description="What is missing or needs improvement")
    gaps: List[str] = Field(description="Key aspects not covered")
    coverage: List[str] = Field(description="Key aspects that ARE covered")


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research quality critic. Evaluate whether the retrieved documents
adequately answer the user's query.

Score guidelines:
- 0.9-1.0: Comprehensive, accurate, well-sourced
- 0.7-0.9: Good coverage with minor gaps
- 0.5-0.7: Partial coverage, key aspects missing
- 0.0-0.5: Poor coverage, major gaps

Respond ONLY with valid JSON matching this schema:
{{
  "score": 0.0-1.0,
  "feedback": "specific gaps to address in next search",
  "gaps": ["gap1", "gap2"],
  "coverage": ["covered1", "covered2"]
}}"""),
    ("human", """Query: {query}

Retrieved Documents:
{documents}

Evaluate and respond with JSON only.""")
])


def run_critic(query: str, research_results: List[str]) -> Tuple[float, str]:
    if not research_results:
        return 0.0, "No documents retrieved — try different search terms"

    docs_text = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc[:500]}"
        for i, doc in enumerate(research_results[:5])
    )

    try:
        response = llm.invoke(
            CRITIC_PROMPT.format_messages(query=query, documents=docs_text)
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        output = CritiqueOutput(**data)
        print(f"[Critic] Score: {output.score:.2f} | Gaps: {output.gaps}")
        return output.score, output.feedback
    except Exception as e:
        print(f"[Critic] Parse error: {e} — defaulting to low score")
        return 0.3, "Could not parse results — retry with broader search terms"
