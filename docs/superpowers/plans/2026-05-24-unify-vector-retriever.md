# Unify Retrieval Around VectorRetriever — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop `MinutesRetriever` and make `VectorRetriever` the single retriever class for every collection (gazete, tutanak, onerge, future).

**Architecture:** `VectorRetriever` already handles everything `MinutesRetriever` did **except** the BM25 hybrid path (deliberately dropped — vector + filter_extractor + reranker is the chosen stack) and the SQLite full-document display (TBMM will now display chunk-level like press). The `year` / `party` / `speaker` kwargs that `MinutesRetriever` exposed to MCP callers are translated to a Chroma `where_filter` dict inside `src/mcp/minutes_server.py` before delegating to `VectorRetriever`.

**Tech Stack:** Python 3.11+, pytest, ChromaDB (embedded), FastAPI, MCP SDK, ruff (project lint default).

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `src/mcp/minutes_server.py` | Modify | Build `where_filter` from year/party/speaker; use `VectorRetriever` |
| `src/retriever/minutes_retriever.py` | Delete | Replaced by `VectorRetriever` |
| `tests/test_minutes_server.py` | Create | Unit tests for `_build_where_filter` + retriever wiring |
| `src/retriever/vector_retriever.py` | Untouched | Already correct |
| `src/retriever/multi_source.py` | Untouched | Already uses `VectorRetriever` |
| `src/generator/service.py` | Untouched | Already uses `VectorRetriever` |
| `src/ui/chat.py` | Untouched | `is_minutes` flag still set by `VectorRetriever` (keyword-derived) |
| `src/evaluator/*` | Untouched | Reads `is_minutes` from `RetrievalResult` — unchanged |
| `tests/test_minutes_sql.py` | Untouched | Pure DB schema smoke tests; not coupled to `MinutesRetriever` |
| `tests/test_query_routing.py` | Untouched | Tests `route_sources()` helper, not coupled |
| `tests/test_vector_retriever.py` | Untouched | Already covers `VectorRetriever` |

## Pre-flight checks

Confirm before starting:
1. Test suite green on `feature/agent-pipeline` head: `python -m pytest tests/ -x -q`
2. No outstanding `MinutesRetriever` consumers besides `src/mcp/minutes_server.py` (grep done — only that one file).

---

## Task 1: Add `_build_where_filter` helper in `minutes_server.py`

**Files:**
- Modify: `src/mcp/minutes_server.py` (add helper, do not yet wire it)
- Create: `tests/test_minutes_server.py`

- [ ] **Step 1: Write failing test for helper**

Create `tests/test_minutes_server.py` with:

```python
"""Tests for src/mcp/minutes_server.py — focuses on where_filter construction."""
from __future__ import annotations

import pytest


class TestBuildWhereFilter:
    """_build_where_filter translates year/party/speaker kwargs to a Chroma where dict."""

    def test_no_filters_returns_none(self):
        from src.mcp.minutes_server import _build_where_filter

        assert _build_where_filter(None, None, None) is None

    def test_year_only_eq(self):
        from src.mcp.minutes_server import _build_where_filter

        assert _build_where_filter(2023, None, None) == {"year": {"$eq": 2023}}

    def test_party_only_eq(self):
        from src.mcp.minutes_server import _build_where_filter

        assert _build_where_filter(None, "CHP", None) == {"party": {"$eq": "CHP"}}

    def test_speaker_only_lowercased(self):
        from src.mcp.minutes_server import _build_where_filter

        assert _build_where_filter(None, None, "Mehmet Yılmaz") == {
            "speaker": {"$eq": "mehmet yılmaz"}
        }

    def test_all_three_combined_with_and(self):
        from src.mcp.minutes_server import _build_where_filter

        result = _build_where_filter(2023, "CHP", "Mehmet Yılmaz")
        assert result == {
            "$and": [
                {"year": {"$eq": 2023}},
                {"party": {"$eq": "CHP"}},
                {"speaker": {"$eq": "mehmet yılmaz"}},
            ]
        }

    def test_year_and_party_only(self):
        from src.mcp.minutes_server import _build_where_filter

        result = _build_where_filter(2023, "CHP", None)
        assert result == {
            "$and": [
                {"year": {"$eq": 2023}},
                {"party": {"$eq": "CHP"}},
            ]
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_minutes_server.py::TestBuildWhereFilter -v`
Expected: ImportError — `cannot import name '_build_where_filter' from 'src.mcp.minutes_server'`

- [ ] **Step 3: Add helper to `src/mcp/minutes_server.py`**

Add **right after** the `from src.config import settings` line near the top of the file (line ~19):

```python
from src.config.collections import get_default_spec
from src.config.document_types import DocumentType


def _build_where_filter(
    year: int | None,
    party: str | None,
    speaker: str | None,
) -> dict | None:
    """Translate the MCP tool's year/party/speaker args to a Chroma `where` dict.

    Speaker is lowercased to match how the TBMM ingestion stores the `speaker`
    metadata field. Returns None when all inputs are None so VectorRetriever can
    fall back to its own date-parsing path on the raw query.
    """
    conds: list[dict] = []
    if year is not None:
        conds.append({"year": {"$eq": year}})
    if party:
        conds.append({"party": {"$eq": party}})
    if speaker:
        conds.append({"speaker": {"$eq": speaker.lower()}})

    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_minutes_server.py::TestBuildWhereFilter -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_minutes_server.py src/mcp/minutes_server.py
git commit -m "feat(minutes_server): add _build_where_filter helper for VectorRetriever migration"
```

---

## Task 2: Verify TBMM Chroma metadata keys match `_build_where_filter` output

**Files:**
- Read-only verification, no source changes
- May produce a follow-up note if mismatch detected

Background: `MinutesRetriever` historically filtered on the `date_year` metadata key (see deleted `_vector_search` line 97). Modern ingestion writes `year` (see `src/trainer/ingestion/adapters/base.py` `DocumentInput.year`). If the TBMM Chroma collection still has chunks indexed under `date_year`, the new path breaks for year filters. This task confirms the on-disk state.

- [ ] **Step 1: Inspect a TBMM chunk's metadata**

Run (one-off, no commit):

```bash
python - <<'PY'
from src.config.collections import get_default_spec
from src.config.document_types import DocumentType
from src.common.chroma import open_collection

spec = get_default_spec(DocumentType.TUTANAK)
print("Collection:", spec.name, "@", spec.db_path)
_, col = open_collection(spec.db_path, spec.name)
sample = col.get(limit=3, include=["metadatas"])
for i, m in enumerate(sample["metadatas"] or []):
    print(f"--- chunk {i} keys ---")
    for k, v in sorted(m.items()):
        print(f"  {k!r}: {v!r}")
PY
```

Expected output: keys including either `"year"` or `"date_year"`, plus `"party"`, `"speaker"`, etc.

- [ ] **Step 2: Decide**

- If keys include `"year"` → proceed to Task 3 unchanged.
- If keys include `"date_year"` but **not** `"year"` → STOP and run Task 2b below before Task 3.
- If neither → the year filter will silently match nothing; STOP and surface to operator.

- [ ] **Step 2b (only if `date_year` is the actual key): patch `_build_where_filter`**

Edit `src/mcp/minutes_server.py` `_build_where_filter`:

```python
    if year is not None:
        conds.append({"date_year": {"$eq": year}})
```

Update test `test_year_only_eq` to expect `{"date_year": {"$eq": 2023}}` and `test_all_three_combined_with_and` accordingly.

Run: `python -m pytest tests/test_minutes_server.py -v`
Expected: green.

Commit:
```bash
git add src/mcp/minutes_server.py tests/test_minutes_server.py
git commit -m "fix(minutes_server): use date_year metadata key for TBMM year filter"
```

---

## Task 3: Wire `VectorRetriever` into `minutes_server.py`

**Files:**
- Modify: `src/mcp/minutes_server.py` (~lines 18, 26–33, 85–90, 130–135)
- Modify: `tests/test_minutes_server.py` (add wiring tests)

- [ ] **Step 1: Write failing wiring tests**

Append to `tests/test_minutes_server.py`:

```python
from unittest.mock import MagicMock, patch


class TestRetrieverWiring:
    """minutes_server uses VectorRetriever with the TBMM spec and translated filters."""

    def _reset_singleton(self):
        import src.mcp.minutes_server as ms

        ms._retriever = None

    def test_get_retriever_uses_vector_retriever_with_tutanak_spec(self):
        self._reset_singleton()
        with patch("src.mcp.minutes_server.VectorRetriever") as mock_vr, \
             patch("src.mcp.minutes_server.get_default_spec") as mock_get_spec:
            from src.mcp.minutes_server import _get_retriever
            from src.config.document_types import DocumentType

            mock_spec = MagicMock()
            mock_get_spec.return_value = mock_spec

            r = _get_retriever()

            mock_get_spec.assert_called_once_with(DocumentType.TUTANAK)
            mock_vr.assert_called_once_with(mock_spec)
            assert r is mock_vr.return_value

    def test_get_retriever_is_singleton(self):
        self._reset_singleton()
        with patch("src.mcp.minutes_server.VectorRetriever") as mock_vr, \
             patch("src.mcp.minutes_server.get_default_spec"):
            from src.mcp.minutes_server import _get_retriever

            r1 = _get_retriever()
            r2 = _get_retriever()
            assert r1 is r2
            assert mock_vr.call_count == 1

    def test_call_tool_passes_where_filter(self):
        self._reset_singleton()
        import asyncio

        with patch("src.mcp.minutes_server.VectorRetriever") as mock_vr, \
             patch("src.mcp.minutes_server.get_default_spec"), \
             patch("src.mcp.minutes_server.build_context", return_value="ctx"), \
             patch("src.mcp.minutes_server.build_structured_context", return_value=[]), \
             patch("src.mcp.minutes_server.format_response", return_value="resp"):
            from src.mcp.minutes_server import call_tool

            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = {
                "documents": [[]], "metadatas": [[]], "distances": [[]],
                "is_minutes": True, "parsed_dates": {},
                "expanded_query": None, "fallback_level": None,
            }
            mock_vr.return_value = mock_instance

            asyncio.run(call_tool(
                "search_parliament_minutes",
                {"query": "ekonomi", "year": 2023, "party": "CHP"},
            ))

            call_kwargs = mock_instance.retrieve.call_args[1]
            assert call_kwargs["where_filter"] == {
                "$and": [
                    {"year": {"$eq": 2023}},
                    {"party": {"$eq": "CHP"}},
                ]
            }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_minutes_server.py::TestRetrieverWiring -v`
Expected: failures — `VectorRetriever` not imported in `minutes_server`, `_get_retriever` still constructs `MinutesRetriever()`.

- [ ] **Step 3: Replace import + `_get_retriever`**

In `src/mcp/minutes_server.py`, replace lines 18, 26-33:

```python
# Old:
from src.retriever.minutes_retriever import MinutesRetriever
# ...
_retriever: Optional[MinutesRetriever] = None


def _get_retriever() -> MinutesRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MinutesRetriever()
    return _retriever
```

With:

```python
from src.retriever.vector_retriever import VectorRetriever

_retriever: Optional[VectorRetriever] = None


def _get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        spec = get_default_spec(DocumentType.TUTANAK)
        _retriever = VectorRetriever(spec)
    return _retriever
```

The `get_default_spec` / `DocumentType` imports were already added in Task 1 Step 3. Verify they are present near line 19; if not, add:

```python
from src.config.collections import get_default_spec
from src.config.document_types import DocumentType
```

- [ ] **Step 4: Update `call_tool` to translate filters**

In `src/mcp/minutes_server.py`, replace the `results = _get_retriever().retrieve(...)` block inside `call_tool` (lines ~85–90):

```python
# Old:
results = _get_retriever().retrieve(
    query,
    year=arguments.get("year"),
    party=arguments.get("party"),
    speaker=arguments.get("speaker"),
)
```

With:

```python
where_filter = _build_where_filter(
    arguments.get("year"),
    arguments.get("party"),
    arguments.get("speaker"),
)
results = _get_retriever().retrieve(query, where_filter=where_filter)
```

- [ ] **Step 5: Update `api_search` the same way**

In `src/mcp/minutes_server.py`, replace the `results = _get_retriever().retrieve(...)` block inside `api_search` (lines ~130–135):

```python
# Old:
results = _get_retriever().retrieve(
    req.query,
    year=req.year,
    party=req.party,
    speaker=req.speaker,
)
```

With:

```python
where_filter = _build_where_filter(req.year, req.party, req.speaker)
results = _get_retriever().retrieve(req.query, where_filter=where_filter)
```

- [ ] **Step 6: Run wiring tests to verify they pass**

Run: `python -m pytest tests/test_minutes_server.py -v`
Expected: all green (Task 1 + Task 3 tests).

- [ ] **Step 7: Run full test suite — guard against regressions**

Run: `python -m pytest tests/ -x -q`
Expected: same pass count as pre-flight baseline. No new failures.

- [ ] **Step 8: Commit**

```bash
git add src/mcp/minutes_server.py tests/test_minutes_server.py
git commit -m "refactor(minutes_server): use VectorRetriever instead of MinutesRetriever

Translate year/party/speaker MCP args into a Chroma where_filter inside
the server; TBMM minutes now use the same retrieval class as press and
onerge. Public MCP / FastAPI request schemas are unchanged."
```

---

## Task 4: Delete `MinutesRetriever`

**Files:**
- Delete: `src/retriever/minutes_retriever.py`

- [ ] **Step 1: Confirm no remaining importers**

Run:

```bash
grep -rn "MinutesRetriever\|from src.retriever.minutes_retriever\|import minutes_retriever" \
  src/ tests/ devtools/ scripts/ 2>/dev/null
```

Expected: zero matches (Task 3 removed the last import).

If anything appears, STOP, fix that file, and re-run the grep before deleting.

- [ ] **Step 2: Delete the file**

```bash
git rm src/retriever/minutes_retriever.py
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: green, same pass count as Task 3 Step 7.

- [ ] **Step 4: Smoke-check the chat UI loads (no live retrieval required)**

Run:

```bash
DEBUG_RAG=0 python -c "from src.ui.chat import run; print('chat module loads')"
```

Expected stdout: `chat module loads`. No `ImportError`.

- [ ] **Step 5: Smoke-check the MCP minutes server loads**

Run:

```bash
python -c "from src.mcp.minutes_server import app, mcp, _build_where_filter; print('minutes_server loads')"
```

Expected stdout: `minutes_server loads`. No `ImportError`.

- [ ] **Step 6: Commit**

```bash
git add -A src/retriever/minutes_retriever.py
git commit -m "refactor(retriever): remove MinutesRetriever; VectorRetriever is the single retriever

All collections (gazete, tutanak, onerge) now use the same retrieval
class. Hybrid BM25 path is intentionally dropped — vector + reranker +
filter_extractor is the chosen stack. TBMM minutes display moves to
chunk-level (same as press), since late chunking already embeds session
context into each chunk."
```

---

## Task 5: Manual UI smoke test (one-time, no commit)

Verify that TBMM queries still return useful results end-to-end. Skip if you cannot run Ollama locally; mark as TODO in the PR description.

- [ ] **Step 1: Start chat**

```bash
source .venv/bin/activate
python chat.py
```

- [ ] **Step 2: Issue a TBMM query through `/müfettiş`**

In the chat REPL, type:

```
/müfettiş 2023 yılında ekonomi politikası
```

Expected:
- No traceback
- Results display with TBMM chunks prefixed with `Kaynak: TBMM | ...` (chunk-level, not full-doc — this is the deliberate semantic change)
- `🔍 Tutanaklar özel filtresi ile arandı.` banner still appears when query has minutes keywords (driven by `RetrievalResult.is_minutes`, computed from keywords in `VectorRetriever`).

- [ ] **Step 3: Issue a press query**

```
ekrem imamoğlu istanbul
```

Expected: gazete chunks, same prefix style as before.

- [ ] **Step 4: Note any regressions**

If TBMM result quality dropped noticeably (missing context that used to come from full-session SQLite content), record it in the PR description. The fix is a follow-up (optional `display_from_sqlite` per-spec flag), **not** part of this plan.

---

## Self-review checklist

- [x] **Spec coverage**: every requirement from the conversation (drop MinutesRetriever, keep VectorRetriever name, no BM25, chunk-level TBMM display, preserve MCP request schema) maps to a task.
- [x] **No placeholders**: every step contains the exact code, command, or expected output.
- [x] **Type consistency**: `_build_where_filter` signature consistent in tests + impl (`int | None`, `str | None`); `VectorRetriever.retrieve(query, *, where_filter=...)` matches the existing signature in `src/retriever/vector_retriever.py:26-34`; metadata key (`year` vs `date_year`) decided in Task 2 before being baked in.
- [x] **Reversibility**: Task 4 is the only destructive step; it lives behind a green test suite + explicit grep gate.

---

## Out of scope (deliberate)

- BM25 / SQLite FTS5 for any collection — revisit only if retrieval quality metrics show exact-match failures.
- `RetrievalResult.is_minutes` migration to per-chunk `doc_type` — keep the field; consumers (chat UI, evaluator) untouched.
- Legacy `src/trainer/press_clips/` removal — separate cleanup task.
- Adapter consolidation (`tutanak_pdf` + `pdf_report` + `kanun_teklifi` → one `DoclingPdfAdapter`) — separate trainer-side plan.
- `vector_search.py` + `vector_retriever.py` merge — cosmetic, defer.
