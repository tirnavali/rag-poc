"""LLM yanıtlarından JSON çıkarma ve Pydantic schema doğrulama yardımcıları."""
import json
import re
from typing import Type, TypeVar, Any

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def extract_json_from_text(text: str) -> str:
    """
    LLM yanıtından JSON içeriğini çıkarır.

    Önce markdown bloğu (```json ... ```) arar; bulamazsa
    metindeki ilk '{' veya '[' ile son '}' veya ']' arasını döndürür.
    Hiçbir eşleşme yoksa orijinal string'i döndürür.
    """
    text = text.strip()

    # Önce markdown bloğu ara
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Markdown bloğu yok; ham JSON olabilir
    # İlk açılış ve son kapanış karakterlerini bul
    start = min((text.find('{') if text.find('{') != -1 else len(text)),
                (text.find('[') if text.find('[') != -1 else len(text)))

    end = max(text.rfind('}'), text.rfind(']'))

    if start < len(text) and end != -1 and end >= start:
        return text[start:end+1]

    # Fallback: orijinal string'i döndür, json.loads'a bırak
    return text

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
