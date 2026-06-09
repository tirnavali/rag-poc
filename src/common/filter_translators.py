"""RAG filtre kriterlerini farklı veritabanı sorgu formatlarına dönüştüren çevirici (translator) katmanı.

Bu modül, arama motorunun ve filtre çıkarıcının veri tabanından (dbsource) bağımsız
çalışabilmesini sağlar. İleride veri tabanı değiştiğinde (örn. Qdrant, Elasticsearch,
Postgres/pgvector), sadece ilgili çevirici sınıfın eklenmesi yeterlidir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from src.common.schemas import FilterCriteria

if TYPE_CHECKING:
    from src.config.document_types import DocumentType


def mask_filters(criteria: FilterCriteria, doc_type: "DocumentType") -> FilterCriteria:
    """Null out filter fields that don't apply to a collection's document type.

    A single FilterCriteria extracted from the query gets applied to every
    selected collection; without masking, a press-only field like `source_name`
    leaks onto parliament collections (and `period`/`session` onto gazete),
    over-filtering them to zero results. Returns a copy with inapplicable fields
    set to None. Unknown/custom types (applicability None) are returned
    unchanged (fail-open).
    """
    from src.config.document_types import FILTER_APPLICABILITY

    allowed = FILTER_APPLICABILITY.get(doc_type)
    if allowed is None:
        return criteria
    drop = {
        field: None
        for field in criteria.model_dump(exclude_none=True)
        if field not in allowed
    }
    return criteria.model_copy(update=drop) if drop else criteria


def build_chroma_where(
    filters: Optional[FilterCriteria],
    collection_key: str,
    *,
    resolver=None,
) -> Optional[dict]:
    """Build a ChromaDB where-filter, resolving `author` against the collection.

    Like ``ChromaFilterTranslator().translate`` but, when an `author` is present,
    first resolves it to the collection's actual labels (case-insensitive,
    Turkish-aware token match) and emits `author $in [labels]`:
      - matches found  → precise `$in` over the titled labels
        (e.g. "Recep Tayyip Erdoğan" → ["BAŞBAKAN RECEP TAYYİP ERDOĞAN"]);
      - vocab present, no match → author dropped (rely on semantic search);
      - vocab unavailable (resolver returns None) → default $eq/$or translation
        (keeps offline/legacy behavior for unknown collections).
    """
    if filters is None:
        return None
    if resolver is None:
        from src.common.author_resolver import AUTHOR_RESOLVER
        resolver = AUTHOR_RESOLVER

    author_in = None
    if filters.author:
        labels = resolver.resolve(collection_key, filters.author)
        if labels is not None:  # vocab available → [] drops, [..] → $in
            author_in = labels
    return ChromaFilterTranslator().translate(filters, author_in=author_in)


class BaseFilterTranslator(ABC):
    """Farklı veritabanları için filtre çeviricilerin türeyeceği soyut temel sınıf."""

    @abstractmethod
    def translate(self, filters: FilterCriteria) -> Optional[Any]:
        """Filtre kriterlerini hedef veritabanının sorgu formatına çevirir.

        Args:
            filters: Çıkarılan filtre kriterlerini içeren FilterCriteria nesnesi.

        Returns:
            Veritabanına özgü filtre sözlüğü/nesnesi veya filtre yoksa None.
        """
        pass


class ChromaFilterTranslator(BaseFilterTranslator):
    """FilterCriteria kriterlerini ChromaDB'nin 'where' filtresi formatına çeviren sınıf."""

    def translate(
        self,
        filters: FilterCriteria,
        *,
        author_in: Optional[list] = None,
    ) -> Optional[dict]:
        """Filtre kriterlerini ChromaDB 'where' koşulu formatına çevirir.

        Tek bir koşul varsa doğrudan o koşulu, birden fazla koşul varsa
        '$and' operatörü ile birleştirilmiş halini döner.

        Args:
            filters: Çıkarılan filtre kriterlerini içeren FilterCriteria nesnesi.
            author_in: Verilirse `filters.author`'ın yerini alır. AuthorResolver
                ile koleksiyondan çözülen tam etiket listesidir; `author $in [...]`
                koşuluna çevrilir. Boş liste → author koşulu hiç eklenmez (filtre
                düşürülür, semantiğe bırakılır). ``None`` → eski $eq/$or davranışı.

        Returns:
            ChromaDB uyumlu where filtre sözlüğü veya filtre kriteri yoksa None.
        """
        conditions = []

        if filters.year is not None:
            conditions.append({"year": {"$eq": filters.year}})

        if filters.year_lte is not None:
            conditions.append({"year": {"$lte": filters.year_lte}})

        if filters.year_gte is not None:
            conditions.append({"year": {"$gte": filters.year_gte}})

        if author_in is not None:
            # Resolver yolu: yalnızca eşleşen etiketler varsa $in ekle; boşsa
            # author koşulu eklenmez (semantik aramaya bırakılır).
            if author_in:
                conditions.append({"author": {"$in": list(author_in)}})
        elif filters.author is not None:
            orig_author = filters.author.strip()
            up_author = orig_author.replace("i", "İ").upper()
            if orig_author != up_author:
                conditions.append({
                    "$or": [
                        {"author": {"$eq": orig_author}},
                        {"author": {"$eq": up_author}}
                    ]
                })
            else:
                conditions.append({"author": {"$eq": orig_author}})

        if filters.author_role is not None:
            conditions.append({"author_role": {"$eq": filters.author_role}})

        if filters.source_name is not None:
            conditions.append({"source_name": {"$eq": filters.source_name}})

        if filters.period is not None:
            conditions.append({"period": {"$eq": filters.period}})

        if filters.session is not None:
            conditions.append({"session": {"$eq": filters.session}})

        if filters.document_type is not None:
            conditions.append({"document_type": {"$eq": filters.document_type}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
