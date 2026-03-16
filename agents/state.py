from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    research_notes: str       # researcher agent output
    critique: str             # critic agent output
    final_answer: str         # synthesizer output
    iteration: int
    messages: Annotated[list, operator.add]
