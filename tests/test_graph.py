import pytest
from unittest.mock import patch, MagicMock
from agents.graph import build_graph


@patch("agents.researcher.hybrid_search")
@patch("agents.researcher.llm")
@patch("agents.critic.llm")
@patch("agents.synthesizer.llm")
def test_full_pipeline(mock_synth, mock_critic, mock_researcher, mock_search):
    mock_search.return_value = [{"id": "1", "score": 0.9, "text": "test doc", "source": "test.txt"}]
    mock_researcher.invoke.return_value = MagicMock(content="Research notes here")
    mock_critic.invoke.return_value = MagicMock(content="PASS")
    mock_synth.invoke.return_value = MagicMock(content="Final synthesized answer")

    graph = build_graph()
    result = graph.invoke({
        "query": "What is RAG?",
        "retrieved_docs": [],
        "research_notes": "",
        "critique": "",
        "final_answer": "",
        "iteration": 0,
        "messages": []
    })

    assert result["final_answer"] == "Final synthesized answer"
    assert result["iteration"] == 1
