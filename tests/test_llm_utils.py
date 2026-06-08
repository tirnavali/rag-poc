import pytest
from pydantic import BaseModel, ValidationError
from src.common.llm_utils import extract_json_from_text, parse_llm_response

class DummySchema(BaseModel):
    name: str
    age: int

def test_extract_json_from_text_markdown():
    text = """Here is the result:
```json
{"name": "Ali", "age": 30}
```
Have a good day!"""
    assert extract_json_from_text(text) == '{"name": "Ali", "age": 30}'

def test_extract_json_from_text_raw_with_text():
    text = "Sure! Here it is: {\"name\": \"Ayse\", \"age\": 25} Let me know if you need anything else."
    assert extract_json_from_text(text) == '{"name": "Ayse", "age": 25}'

def test_extract_json_from_text_just_json():
    text = '{"name": "Veli", "age": 40}'
    assert extract_json_from_text(text) == '{"name": "Veli", "age": 40}'

def test_parse_llm_response_success():
    text = "```json\n{\"name\": \"Ahmet\", \"age\": 50}\n```"
    result = parse_llm_response(text, DummySchema)
    assert isinstance(result, DummySchema)
    assert result.name == "Ahmet"
    assert result.age == 50

def test_parse_llm_response_invalid_json():
    text = "```json\n{\"name\": \"Ahmet\", \"age\": }\n```" # Malformed JSON
    with pytest.raises(ValueError, match="Failed to decode JSON"):
        parse_llm_response(text, DummySchema)

def test_parse_llm_response_validation_error():
    text = "```json\n{\"name\": \"Ahmet\", \"age\": \"elli\"}\n```" # Wrong type
    with pytest.raises(ValidationError):
        parse_llm_response(text, DummySchema)

