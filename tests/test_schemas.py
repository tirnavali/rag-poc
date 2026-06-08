import pytest
from pydantic import ValidationError
from src.common.schemas import Message, ToolCall, ToolCallFunction, SearchResult, RoutingDecision

def test_message_valid():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.name is None

def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="Hello")

def test_rag_base_model_extra_forbid():
    # RAGBaseModel (inherited by Message) should forbid extra fields
    with pytest.raises(ValidationError):
        Message(role="user", content="Hello", extra_field="not allowed")

def test_tool_call_valid():
    func = ToolCallFunction(name="search", arguments='{"query": "test"}')
    tc = ToolCall(id="call_123", function=func)
    assert tc.id == "call_123"
    assert tc.function.name == "search"
    assert tc.type == "function"

def test_search_result_valid():
    res = SearchResult(document="Some text", distance=0.1, metadata={"source": "test"})
    assert res.document == "Some text"
    assert res.distance == 0.1
    assert res.metadata["source"] == "test"

def test_search_result_missing_required():
    with pytest.raises(ValidationError):
        SearchResult(document="Missing distance")

def test_routing_decision_valid():
    decision = RoutingDecision(intents=["minutes"], reasoning="Query about parliament")
    assert decision.intents == ["minutes"]
    assert "parliament" in decision.reasoning

def test_validate_assignment():
    msg = Message(role="user", content="Hello")
    # This should trigger validation and fail because 'role' must be one of the Literals
    with pytest.raises(ValidationError):
        msg.role = "bad_role"
