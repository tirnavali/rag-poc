"""LLM yanıtlarından JSON çıkarma ve Pydantic schema doğrulama yardımcıları."""
import json
import re
from typing import Type, TypeVar, Any

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def repair_json_brackets(text: str) -> str:
    """
    JSON string'indeki eşleşmeyen veya kapatılmamış parantezleri ({}, []) onarır.
    """
    repaired = []
    stack = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
                repaired.append(char)
            elif char == '\\':
                escaped = True
                repaired.append(char)
            elif char == '"':
                in_string = False
                repaired.append(char)
            else:
                repaired.append(char)
        else:
            if char == '"':
                in_string = True
                repaired.append(char)
            elif char in ('{', '['):
                stack.append(char)
                repaired.append(char)
            elif char == '}':
                while stack and stack[-1] == '[':
                    repaired.append(']')
                    stack.pop()
                if stack and stack[-1] == '{':
                    stack.pop()
                repaired.append(char)
            elif char == ']':
                while stack and stack[-1] == '{':
                    repaired.append('}')
                    stack.pop()
                if stack and stack[-1] == '[':
                    stack.pop()
                repaired.append(char)
            else:
                repaired.append(char)

    while stack:
        open_char = stack.pop()
        if open_char == '{':
            repaired.append('}')
        elif open_char == '[':
            repaired.append(']')

    return "".join(repaired)


def extract_json_from_text(text: str) -> str:
    """
    LLM yanıtından JSON içeriğini çıkarır ve bozuk parantezleri onarır.

    Önce markdown bloğu (```json ... ```) arar; bulamazsa
    metindeki ilk '{' veya '[' ile son '}' veya ']' arasını döndürür.
    Hiçbir eşleşme yoksa orijinal string'i döndürür.
    """
    text = text.strip()

    # Önce markdown bloğu ara
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
    else:
        # Markdown bloğu yok; ham JSON olabilir
        # İlk açılış ve son kapanış karakterlerini bul
        start = min((text.find('{') if text.find('{') != -1 else len(text)),
                    (text.find('[') if text.find('[') != -1 else len(text)))

        end = max(text.rfind('}'), text.rfind(']'))

        if start < len(text) and end != -1 and end >= start:
            extracted = text[start:end+1]
        else:
            extracted = text

    return repair_json_brackets(extracted)

def parse_llm_response(response: str, schema: Type[T]) -> T:
    """
    Ham LLM yanıtından JSON çıkarır ve verilen Pydantic schema'sına karşı doğrular.

    Raises:
        ValueError: JSON parse edilemezse.
        ValidationError: JSON, schema ile uyuşmazsa.
    """
    json_str = extract_json_from_text(response)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from LLM response. Extracted string: {json_str[:100]}... Error: {e}")

    return schema.model_validate(data)
