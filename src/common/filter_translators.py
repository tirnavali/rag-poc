"""RAG filtre kriterlerini farklı veritabanı sorgu formatlarına dönüştüren çevirici (translator) katmanı.

Bu modül, arama motorunun ve filtre çıkarıcının veri tabanından (dbsource) bağımsız
çalışabilmesini sağlar. İleride veri tabanı değiştiğinde (örn. Qdrant, Elasticsearch,
Postgres/pgvector), sadece ilgili çevirici sınıfın eklenmesi yeterlidir.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.common.schemas import FilterCriteria


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

    def translate(self, filters: FilterCriteria) -> Optional[dict]:
        """Filtre kriterlerini ChromaDB 'where' koşulu formatına çevirir.

        Tek bir koşul varsa doğrudan o koşulu, birden fazla koşul varsa
        '$and' operatörü ile birleştirilmiş halini döner.

        Args:
            filters: Çıkarılan filtre kriterlerini içeren FilterCriteria nesnesi.

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

        if filters.author is not None:
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
