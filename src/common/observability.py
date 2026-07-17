"""Langfuse gözlemlenebilirlik fabrikası.

Tek modül istemciyi ve LangChain handler'ını sahiplenir; geri kalan kod
yalnız buradan import eder. LANGFUSE_ENABLED=0 iken her fabrika None döndürür
— SDK nesnesi hiç yaratılmaz, pipeline'a sıfır yük biner. SDK arka plan
thread'inde batch'lediği ve başarısız export'ları sessizce düşürdüğü için
Langfuse sunucusunun ölü olması pipeline'ı asla yavaşlatmaz/bozmaz.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.config import settings

_logger = logging.getLogger(__name__)

_client: Any = None


def get_langfuse() -> Optional[Any]:
    """Singleton Langfuse istemcisi; kapalıyken veya SDK yokken None."""
    global _client
    if not settings.LANGFUSE_ENABLED:
        return None
    if _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse(
                base_url=settings.LANGFUSE_BASE_URL,
                # Ölü sunucu pipeline'ı bekletmesin diye kısa HTTP timeout.
                timeout=5,
            )
        except Exception as e:
            _logger.warning("Langfuse istemcisi başlatılamadı (izleme kapalı): %s", e)
            return None
    return _client


def get_langchain_handler() -> Optional[Any]:
    """graph.invoke(config={'callbacks': [handler]}) için CallbackHandler."""
    if get_langfuse() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as e:
        _logger.warning("Langfuse CallbackHandler oluşturulamadı: %s", e)
        return None


def shutdown_langfuse() -> None:
    """Tamponlanmış span'leri flush eder — FastAPI lifespan teardown'dan çağrılır.

    (CLI çıkışını SDK'nın kendi atexit kancası kapatır.)
    """
    global _client
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass
        _client = None
