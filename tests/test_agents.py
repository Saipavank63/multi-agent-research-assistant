import pytest
from unittest.mock import patch, MagicMock
from agents.critic import run_critic
from agents.synthesizer import run_synthesizer


def test_critic_returns_low_score_for_empty_results():
    score, feedback = run_critic("What is LangGraph?", [])
    assert score == 0.0
    assert len(feedback) > 0


def test_critic_score_in_valid_range():
    mock_docs = [
        "LangGraph is a framework for building stateful multi-agent systems.",
        "It extends LangChain with graph-based state machines.",
    ]
    with patch("agents.critic.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.85, "feedback": "", "gaps": [], "coverage": ["LangGraph basics"]}'
        )
        score, _ = run_critic("What is LangGraph?", mock_docs)
        assert 0.0 <= score <= 1.0


def test_synthesizer_handles_empty_results():
    result = run_synthesizer("What is RAG?", [])
    assert "Insufficient" in result or len(result) > 0


def test_synthesizer_returns_string():
    mock_docs = ["RAG stands for Retrieval Augmented Generation."]
    with patch("agents.synthesizer.llm") as mock_llm:
        mock_llm.invoke.return_value = MagicMock(
            content="RAG is a technique that combines retrieval with generation."
        )
        result = run_synthesizer("What is RAG?", mock_docs)
        assert isinstance(result, str)
        assert len(result) > 0


def test_research_state_structure():
    from agents.orchestrator import ResearchState
    required_fields = [
        "query", "messages", "research_results",
        "critique_score", "critique_feedback", "final_answer", "iteration_count"
    ]
    for field in required_fields:
        assert field in ResearchState.__annotations__, f"Missing field: {field}"
