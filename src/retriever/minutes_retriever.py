"""Parliament-minutes-specific retriever.

Uses only the TBMM minutes collection. Accepts explicit year/party/speaker
filters so callers (MCP tools, evaluator) don't rely on keyword routing.
"""
from __future__ import annotations

from typing import Optional

from src.common.chroma import open_collection
from src.common.dates import extract_dates
from src.common.embeddings import build_embedder
from src.common.protocols import RetrievalResult
from src.common.sqlite_io import connect
from src.common.text import extract_relevant_windows, normalize_tr
from src.config import settings


class MinutesRetriever:
    def __init__(self) -> None:
        self.embeddings = build_embedder()
        _, self.collection = open_collection(
            settings.MINUTES_CHROMA, settings.MINUTES_COLLECTION
        )
        self.conn = connect(settings.MINUTES_SQLITE)
        self.cur = self.conn.cursor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_year(self, query: str, explicit_year: Optional[int]) -> list[int]:
        if explicit_year is not None:
            return [explicit_year]
        parsed = extract_dates(query)
        exact_dates = parsed.get("exact_dates", [])
        year_list = parsed.get("years", [])
        year_from_exact = [int(d[:4]) for d in exact_dates if d]
        return list(set([int(y) for y in year_list] + year_from_exact))

    def _fts_search(
        self,
        fts_query: str,
        years: list[int],
        party: Optional[str],
        speaker: Optional[str],
    ) -> dict[str, int]:
        """BM25 FTS5 search. Year filters on SQL; party/speaker boost via query terms."""
        bm25_ranks: dict[str, int] = {}
        if not fts_query:
            return bm25_ranks

        # Prepend party/speaker as extra search terms for BM25 boost
        extra_terms = " ".join(
            normalize_tr(t) for t in [party, speaker] if t
        )
        effective_query = f"{extra_terms} {fts_query}".strip() if extra_terms else fts_query

        params: list = [effective_query]
        extra = ""

        if years:
            placeholders = " OR ".join(["m.date LIKE ?"] * len(years))
            extra += f" AND ({placeholders})"
            params.extend(f"{y}%" for y in years)

        sql = f"""
            SELECT f.rowid, bm25(f.parliament_minutes_fts) AS score
            FROM parliament_minutes_fts f
            JOIN parliament_minutes m ON f.rowid = m.id
            WHERE f.parliament_minutes_fts MATCH ? {extra}
            ORDER BY bm25(f.parliament_minutes_fts) ASC
            LIMIT {settings.FTS_LIMIT}
        """
        try:
            self.cur.execute(sql, params)
            for rank, row in enumerate(self.cur.fetchall(), 1):
                bm25_ranks[str(row[0])] = rank
        except Exception as e:
            print(f"MinutesRetriever FTS error: {e}")
        return bm25_ranks

    def _vector_search(
        self,
        q_vec: list[float],
        fetch_k: int,
        years: list[int],
        party: Optional[str],
        speaker: Optional[str],
    ) -> dict[str, dict]:
        """ChromaDB vector search with date_year + optional party/speaker AND filters."""
        chroma_best: dict[str, dict] = {}
        try:
            and_conditions: list[dict] = []

            if years:
                year_conditions = [{"date_year": {"$eq": y}} for y in years]
                if len(year_conditions) == 1:
                    and_conditions.append(year_conditions[0])
                else:
                    and_conditions.append({"$or": year_conditions})

            if party:
                and_conditions.append({"party": {"$eq": party}})

            if speaker:
                and_conditions.append({"speaker": {"$eq": speaker.lower()}})

            if not and_conditions:
                where_filter = None
            elif len(and_conditions) == 1:
                where_filter = and_conditions[0]
            else:
                where_filter = {"$and": and_conditions}

            opts: dict = {
                "query_embeddings": [q_vec],
                "n_results": fetch_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                opts["where"] = where_filter

            res = self.collection.query(**opts)
            for i, cid in enumerate(res.get("ids", [[]])[0]):
                try:
                    k_no = int(cid.split("_")[0])
                except Exception:
                    continue
                key = str(k_no)
                if key not in chroma_best:
                    chroma_best[key] = {
                        "rank": len(chroma_best) + 1,
                        "dist": res["distances"][0][i],
                        "doc": res["documents"][0][i],
                        "meta": res["metadatas"][0][i],
                    }
        except Exception as e:
            print(f"MinutesRetriever Chroma error: {e}")
        return chroma_best

    def _fetch_row(self, k_no: int, query: str) -> Optional[tuple]:
        self.cur.execute(
            "SELECT term, legislative_year, session, date, speaker, content "
            "FROM parliament_minutes WHERE id = ?",
            (k_no,),
        )
        row = self.cur.fetchone()
        if not row:
            return None
        term, legislative_year, session, date, speaker, metin = row
        title = f"TBMM Term: {term}, Legislative Year: {legislative_year}, Session: {session}"
        yazar_m = speaker or "Unknown Speaker"
        prefix = f"Source: TBMM Minutes | Date: {date} | Speaker: {yazar_m} | Session: {title}\n"
        doc = prefix + extract_relevant_windows(str(metin), query)
        meta = {
            "source_db": "minutes",
            "kayit_no": k_no,
            "sqlite_row_id": k_no,
            "publication": "TBMM Minutes",
            "date": str(date)[:10] if date else "?",
            "author": yazar_m,
            "title": title,
            "topics": "",
        }
        return doc, meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        year: Optional[int] = None,
        party: Optional[str] = None,
        speaker: Optional[str] = None,
        top_k: int = settings.RETRIEVE_TOP_K,
        fetch_k: int = settings.RETRIEVE_FETCH_K,
    ) -> RetrievalResult:
        years = self._resolve_year(query, year)
        words = [w for w in normalize_tr(query).split() if w.isalnum()]
        fts_query = " OR ".join(words)
        q_vec = self.embeddings.embed_query(query)

        bm25 = self._fts_search(fts_query, years, party, speaker)
        chroma = self._vector_search(q_vec, fetch_k, years, party, speaker)

        k = settings.RRF_K
        fusion: dict[str, float] = {}
        for key in set(list(bm25) + list(chroma)):
            score = 0.0
            if key in bm25:
                score += 1.0 / (k + bm25[key])
            if key in chroma:
                score += 1.0 / (k + chroma[key]["rank"])
            fusion[key] = score

        sorted_keys = sorted(fusion.items(), key=lambda x: x[1], reverse=True)

        final_docs: list[str] = []
        final_metas: list[dict] = []
        final_dists: list[float] = []

        for key, f_score in sorted_keys[:top_k]:
            row = self._fetch_row(int(key), query)
            if row is None:
                continue
            doc, meta = row
            final_docs.append(doc)
            final_metas.append(meta)
            final_dists.append(max(0.0, 1.0 - (f_score * 20)))

        return RetrievalResult(
            documents=[final_docs],
            metadatas=[final_metas],
            distances=[final_dists],
            is_minutes=True,
            parsed_dates={"years": [str(y) for y in years]},
            expanded_query=None,
        )
