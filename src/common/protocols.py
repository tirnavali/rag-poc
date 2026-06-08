"""Typing.Protocol definitions for the Retriever, Generator, and Evaluator layers.

These decouple callers from concrete classes. The TypedDicts mirror the shape
that the UI already consumes (Chroma's list-of-lists convention), so existing
call sites remain compatible during the migration.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Protocol, TypedDict


class RetrievalResult(TypedDict):
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
    is_minutes: bool
    parsed_dates: dict
    expanded_query: Optional[str]
    fallback_level: Optional[str]


class StreamChunk(TypedDict):
    type: Literal["thinking", "content"]
    content: str


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = ...,
        fetch_k: int = ...,
        mufettis_mode: bool = False,
    ) -> RetrievalResult: ...

    def inspect_record(self, source_db: str, chunk_id: str) -> Optional[dict]: ...


class Generator(Protocol):
    def expand_query(self, query: str) -> str: ...

    def answer(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
    ) -> tuple[str, str]: ...

    def stream(
        self,
        query: str,
        context: str,
        *,
        mufettis_mode: bool = False,
    ) -> Iterable[StreamChunk]: ...


class Evaluator(Protocol):
    def evaluate(
        self,
        queries: list[dict],
        retriever: Retriever,
        generator: Generator,
    ) -> Any: ...
