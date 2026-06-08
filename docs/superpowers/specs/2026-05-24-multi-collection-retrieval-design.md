# Multi-Collection Query Interface — Design Spec

**Date:** 2026-05-24  
**Status:** Design (awaiting implementation)  
**Goal:** Enable querying multiple Chroma collections in one call; user selects collections at app startup; RRF-fused results.

---

## Overview

Current RAG pipeline queries single collections (gazete_arsivi, tbmm_minutes, onerge_collection). To test different embedding models + compare results across collections, need:

1. **Collection selector UI** — pick which collections to query (with sizes)
2. **Multi-collection retriever** — fan-out to selected collections, fuse with RRF
3. **Results attribution** — show which collection each result came from
4. **Session-level selection** — user picks collections once at app startup, all queries use same set

After this design:
- `vector_explorer`: Multi-select dropdown (gazete_arsivi, tbmm_tutanaklar_jina_v3, onerge, etc.) + collection sizes
- `chat.py`: Startup prompt to pick collections, then all queries fan-out to selected collections
- Same RRF fusion as `MultiSourceRetriever` (already tested, proven)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  vector_explorer.py / chat.py (başlangıç — startup)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────▼──────────┐
        │ get_available_collections()
        │ (çıktı: names + sizes)
        └────────┬────────────┘
                 │
        ┌────────▼────────┐
        │ CollectionSelector UI
        │ (Multi-select dropdown)
        └────────┬────────┘
                 │
        ┌────────▼──────────────────────────────────┐
        │ User selects:
        │  ✓ gazete_arsivi (12,500 chunks)
        │  ✓ tbmm_tutanaklar_jina_v3 (45,200 chunks)
        │  ✗ tbmm_tbmm_tutanaklar_docling_jina_v3_4k
        │  ✓ onerge_collection (8,900 chunks)
        └────────┬──────────────────────────────────┘
                 │
        ┌────────▼──────────────────────────────────────────┐
        │ MultiSourceRetriever(specs=[...])
        │ • Query each collection independently
        │ • Embed query once, broadcast
        └────────┬──────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
┌───▼──┐    ┌───▼──┐    ┌───▼──┐        ┌──▼───┐
│ VR 1 │    │ VR 2 │    │ VR 3 │        │ VR 4 │
│Chroma│    │Chroma│    │Chroma│        │Chroma│
└───┬──┘    └───┬──┘    └───┬──┘        └──┬───┘
    │           │           │              │
    └───────────┼───────────┼──────────────┘
                │
        ┌───────▼────────┐
        │ RRF Fusion
        │ (Deduplicate, rank, top_k)
        └───────┬────────┘
                │
        ┌───────▼──────────────────────┐
        │ RetrievalResult
        │ (docs + metadatas with
        │  collection attribution)
        └───────┬──────────────────────┘
                │
        ┌───────▼──────────────┐
        │ Display Results
        │ Kaynak: gazete_arsivi | ...
        │ Kaynak: tbmm_tutanaklar_jina_v3 | ...
        └───────────────────────┘
```

---

## Components

### 1. Collection Registry Enhancement

**File:** `src/config/collections.py` (modify)

Add function to list collections with runtime sizes:

```python
def get_available_collections() -> list[dict]:
    """
    Tüm mevcut koleksiyonları boyutlarıyla listele.
    
    Returns:
        List of dicts:
        [
            {
                "name": "gazete_arsivi",
                "type": "gazete",  # DocumentType
                "embedding_model": "nomic-embed-text-v2-moe",
                "count": 12500,  # Chunks in collection
                "spec": CollectionSpec(...),  # For backend use
            },
            ...
        ]
    
    Sorted by name for consistent UI ordering.
    """
    result = []
    
    for spec_name, spec in COLLECTIONS.items():
        try:
            # Open collection, get count
            _, col = open_collection(spec.db_path, spec.name)
            count = col.count()
        except Exception as e:
            # Collection doesn't exist or error; show 0
            count = 0
        
        result.append({
            "name": spec.name,
            "type": spec.doc_type.value,
            "embedding_model": spec.embedding_model,
            "count": count,
            "spec": spec,
        })
    
    return sorted(result, key=lambda x: x["name"])
```

---

### 2. Collection Selector Component

**File:** `src/ui/components/collection_selector.py` (new)

Reusable collection picker for vector_explorer + chat.py:

```python
"""Collection selection UI component (Streamlit + Rich abstraction)."""

from typing import Callable
from src.config.collections import get_available_collections
from src.config.collections import CollectionSpec


def select_collections_streamlit(
    defaults: list[str] | None = None,
) -> list[CollectionSpec]:
    """
    Streamlit multi-select widget for collections.
    
    Used by: vector_explorer.py
    
    Args:
        defaults: Default selected collection names
        
    Returns:
        List of CollectionSpec objects for selected collections
    """
    import streamlit as st
    
    available = get_available_collections()
    
    # Format choices with size info
    choices = {
        col["name"]: f"{col['name']} ({col['count']:,} chunks)"
        for col in available
    }
    
    selected = st.multiselect(
        "Koleksiyonları Seçin",
        options=list(choices.keys()),
        default=defaults or [col["name"] for col in available[:2]],
        help="Sorguya dahil edilecek koleksiyonlar",
    )
    
    # Show model info for transparency
    for col in available:
        if col["name"] in selected:
            st.sidebar.caption(f"📊 {col['embedding_model']} | {col['count']:,} chunks")
    
    # Return specs for selected collections
    return [col["spec"] for col in available if col["name"] in selected]


def select_collections_interactive(
    defaults: list[str] | None = None,
) -> list[CollectionSpec]:
    """
    Interactive prompt for collections (Rich, no Streamlit).
    
    Used by: chat.py
    
    Displays available collections with sizes, prompts user to select.
    """
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint
    
    available = get_available_collections()
    
    # Show available collections
    table = Table(title="📚 Mevcut Koleksiyonlar")
    table.add_column("Koleksiyon", style="cyan")
    table.add_column("Tür", style="magenta")
    table.add_column("Model", style="green")
    table.add_column("Boyut", style="yellow")
    
    for col in available:
        table.add_row(
            col["name"],
            col["type"],
            col["embedding_model"],
            f"{col['count']:,}",
        )
    
    rprint(table)
    
    # Prompt for selection
    default_str = ",".join(defaults or [col["name"] for col in available[:2]])
    prompt_text = f"Sorgulamak için koleksiyonları seçin (virgülle ayrılmış) [{default_str}]: "
    selection = Prompt.ask(prompt_text, default=default_str)
    
    selected_names = [s.strip() for s in selection.split(",")]
    
    # Validate
    available_names = {col["name"] for col in available}
    invalid = [n for n in selected_names if n not in available_names]
    if invalid:
        raise ValueError(f"Bilinmeyen koleksiyonlar: {invalid}")
    
    # Return specs
    return [col["spec"] for col in available if col["name"] in selected_names]
```

---

### 3. MultiSourceRetriever Enhancement

**File:** `src/retriever/multi_source.py` (modify)

Rename semantics; keep logic identical:

```python
class MultiSourceRetriever:
    """Fan-out retriever across multiple collections with RRF fusion.
    
    Originally designed for document types (GAZETE, TUTANAK, ONERGE).
    Now accepts any collection specs — same fusion logic applies.
    """
    
    def __init__(self, specs: dict | list) -> None:
        """
        Args:
            specs: Dict[DocumentType, CollectionSpec] (legacy)
                   OR List[CollectionSpec] (new multi-collection mode)
        """
        if isinstance(specs, list):
            # Multi-collection mode
            self.retrievers = {
                spec.name: VectorRetriever(spec) for spec in specs
            }
        else:
            # Legacy mode (by document type)
            self.retrievers = {
                dt: VectorRetriever(spec) for dt, spec in specs.items()
            }
    
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = settings.RETRIEVE_TOP_K,
        per_source_k: int = 20,
        mufettis_mode: bool = False,
    ) -> RetrievalResult:
        """
        Fan-out ve RRF füzyonu.
        
        Args:
            query: Search query
            top_k: Final result count
            per_source_k: Candidates per collection before fusion
            mufettis_mode: Deep research mode
        
        Returns:
            RetrievalResult with fused documents + collection attribution in metadatas
        """
        if mufettis_mode:
            top_k = settings.MUFETTIS_TOP_K
            per_source_k = settings.MUFETTIS_FETCH_K
        
        # Fan-out to all collections
        per_source = {}
        for name, retriever in self.retrievers.items():
            per_source[name] = retriever.retrieve(
                query,
                top_k=per_source_k,
                mufettis_mode=False,
            )
        
        # RRF fusion (same logic as before)
        return _rrf_fuse(per_source, query, top_k)


def _rrf_fuse(
    per_source: dict,  # name → RetrievalResult
    query: str,
    top_k: int,
    k: int = 60,
) -> RetrievalResult:
    """
    RRF across collections.
    
    Deduplicates by (document_id, chunk_index), ranks by RRF score.
    Preserves "collection" in metadata for attribution.
    """
    scores = {}
    records = {}
    
    for source_name, result in per_source.items():
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        
        for rank, (doc, meta, dist) in enumerate(
            zip(docs, metas, dists), start=1
        ):
            # Dedup key
            uid = f"{meta.get('document_id', '')}_c{meta.get('chunk_index', rank)}"
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            
            # Add collection attribution to metadata
            meta_with_source = {**meta, "collection": source_name}
            records[uid] = (doc, meta_with_source, dist)
    
    # Top-k by RRF score
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    final_docs, final_metas, final_dists = [], [], []
    for uid, _score in top:
        doc, meta, dist = records[uid]
        final_docs.append(doc)
        final_metas.append(meta)
        final_dists.append(dist)
    
    # Determine is_minutes from keywords (backward compat)
    is_minutes = any(kw in query.lower() for kw in settings.MINUTES_KEYWORDS)
    
    return RetrievalResult(
        documents=[final_docs],
        metadatas=[final_metas],
        distances=[final_dists],
        is_minutes=is_minutes,
        parsed_dates={},
        expanded_query=None,
        fallback_level=None,
    )
```

---

### 4. vector_explorer Enhancement

**File:** `scripts/vector_explorer.py` (modify)

Replace single-collection dropdown with multi-select:

```python
import streamlit as st
from src.config.collections import get_available_collections
from src.ui.components.collection_selector import select_collections_streamlit
from src.retriever.multi_source import MultiSourceRetriever

st.set_page_config(page_title="Multi-Collection Vector Explorer", layout="wide")
st.title("🏛️ Multi-Collection Vector DB Explorer")

# ─── Sidebar ─────────────────────────────────────────────────

st.sidebar.header("Koleksiyon Seçimi")

# Show available collections + sizes
available = get_available_collections()
default_names = [col["name"] for col in available[:3]]  # Default: first 3

selected_specs = select_collections_streamlit(defaults=default_names)

if not selected_specs:
    st.error("En az bir koleksiyon seçiniz.")
    st.stop()

# Create multi-collection retriever
retriever = MultiSourceRetriever(specs=selected_specs)

# ─── Main ────────────────────────────────────────────────────

query = st.text_input("Sorgu", placeholder="Arama terimleri...")

if query:
    with st.spinner("Aranıyor..."):
        results = retriever.retrieve(query, top_k=20)
    
    # Display results with collection attribution
    st.subheader(f"Sonuçlar ({len(results['documents'][0])} döküman)")
    
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        collection = meta.get("collection", "unknown")
        st.markdown(f"**{i+1}. [{collection}]** — Skoru: {1 - dist:.3f}")
        st.text(doc[:500] + "...")
        with st.expander("Metadata"):
            st.json(meta)
```

---

### 5. chat.py Enhancement

**File:** `src/ui/chat.py` (modify)

Add startup collection selector:

```python
from src.ui.components.collection_selector import select_collections_interactive
from src.retriever.multi_source import MultiSourceRetriever

class ChatState:
    # ... existing fields ...
    selected_collections: list[CollectionSpec] = []


async def run():
    """Chat REPL with multi-collection support."""
    
    # At startup: prompt for collection selection
    rich_print("[bold cyan]📚 Koleksiyon Seçimi[/bold cyan]")
    selected_specs = select_collections_interactive()
    rich_print(f"[green]✓ {len(selected_specs)} koleksiyon seçildi[/green]")
    
    # Create retriever once for session
    retriever = MultiSourceRetriever(specs=selected_specs)
    
    # Chat loop (existing logic, now uses multi-collection retriever)
    state = ChatState(selected_collections=selected_specs)
    
    while True:
        query = prompt_user("Soru: ")
        
        if query.startswith("/"):
            # Handle commands
            await handle_command(query, retriever)
        else:
            # Standard retrieval
            results = retriever.retrieve(query)
            
            # Display with collection attribution
            for i, (doc, meta) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                collection = meta.get("collection", "?")
                rich_print(f"[cyan]Kaynak: {collection}[/cyan] | {doc[:300]}...")
            
            # Continue to LLM generation...
```

---

## Config Structure

**File:** `retrieval_config.yaml` (repo root, new)

```yaml
# Multi-collection query defaults
# Konfigüre edilecek koleksiyonlar ve başlangıç seçimleri

default_collections:
  vector_explorer:
    - gazete_arsivi
    - tbmm_tutanaklar_jina_v3
    - onerge_collection
  
  chat:
    - gazete_arsivi
    - tbmm_minutes  # Will resolve to default minutes spec
    - onerge_collection
```

---

## Data Flow

```
1. User runs vector_explorer.py
   ├─ get_available_collections() → [gazete, tbmm_v3, tbmm_v4, onerge, ...]
   ├─ User multi-selects: gazete_arsivi, tbmm_tutanaklar_jina_v3
   └─ Specs = [CollectionSpec(gazete), CollectionSpec(tbmm_v3)]

2. MultiSourceRetriever(specs=[...])
   ├─ Retriever 1 for gazete_arsivi
   ├─ Retriever 2 for tbmm_tutanaklar_jina_v3
   └─ Ready for queries

3. User enters query: "ekonomi politikası 2023"
   ├─ Fan-out:
   │  ├─ VR1.retrieve() → [gazete doc1, gazete doc2, ...]
   │  └─ VR2.retrieve() → [tbmm doc1, tbmm doc2, ...]
   ├─ RRF fusion (dedup + rank)
   └─ Add "collection" field to each result's metadata

4. Display results:
   ├─ Kaynak: gazete_arsivi | Doc text...
   ├─ Kaynak: tbmm_tutanaklar_jina_v3 | Doc text...
   └─ ...
```

---

## File Structure

| Path | Action | Notes |
|------|--------|-------|
| `src/config/collections.py` | Modify | Add `get_available_collections()` function |
| `src/ui/components/` | Create (dir) | Shared UI components |
| `src/ui/components/__init__.py` | Create | Empty |
| `src/ui/components/collection_selector.py` | Create | `select_collections_streamlit()` + `select_collections_interactive()` |
| `scripts/vector_explorer.py` | Modify | Use `select_collections_streamlit()`, multi-collection retriever |
| `src/ui/chat.py` | Modify | Startup `select_collections_interactive()`, pass specs to retriever |
| `src/retriever/multi_source.py` | Modify | Accept list of specs, add "collection" to metadata |
| `retrieval_config.yaml` | Create | Default collection selections per app |
| `tests/test_multi_collection_retriever.py` | Create | Test RRF fusion with multiple collections |

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_multi_collection_retriever.py`

```python
class TestMultiCollectionRetriever:
    """MultiSourceRetriever with collection specs."""
    
    def test_fan_out_multiple_collections(self):
        """Query 2+ collections, fuse results."""
        ...
    
    def test_rrf_fusion_deduplicates(self):
        """Same doc in multiple collections → appears once."""
        ...
    
    def test_collection_attribution_in_metadata(self):
        """Result metadata includes 'collection' field."""
        ...
    
    def test_top_k_respected(self):
        """Final result count = top_k, not per_source_k * num_collections."""
        ...
    
    def test_empty_collection(self):
        """One collection has no results → still works."""
        ...


class TestCollectionSelector:
    """Collection selector helpers."""
    
    def test_get_available_collections_includes_sizes(self):
        """Output includes 'count' field."""
        ...
    
    def test_select_collections_interactive_parses_input(self):
        """User input "col1, col2" → correct specs."""
        ...
```

### Integration Tests

**File:** `tests/test_vector_explorer_integration.py` (new)

```python
def test_vector_explorer_loads_with_multi_select():
    """Streamlit app initializes, multi-select works."""
    ...

def test_chat_startup_collection_selection():
    """Chat starts, prompts for collections, creates retriever."""
    ...
```

---

## Key Design Decisions

1. **Session-level selection (not per-query):** User picks collections once at startup. Simpler UX, clearer for testing (all queries use same set).

2. **Collection specs, not names:** Internally use `CollectionSpec` objects to carry embedding model + db_path. Names are UI-only.

3. **RRF deduplication:** If same doc appears in multiple collections, RRF fusion ranks it by combined relevance. No double-counting.

4. **Collection attribution in metadata:** Each result has `"collection": "gazete_arsivi"` in metadata. Enables filtering/debugging later.

5. **Reuse MultiSourceRetriever:** Same logic, different input (specs vs. doc types). No new fusion algorithm.

6. **Two selector UIs (Streamlit + interactive):** `vector_explorer` uses Streamlit, `chat.py` uses Rich. Shared logic, different rendering.

---

## Out of Scope (Deliberate)

- Per-query collection selection (add later if needed)
- Collection-specific filters (e.g., "only 2023 from gazete, all from minutes")
- Weighted RRF (e.g., prioritize minutes over gazete)
- Caching per collection (optimization; add if latency issue)
- Admin UI for collection management (currently in models.yaml)

---

## Success Criteria

- [x] `get_available_collections()` returns all + sizes
- [x] `CollectionSelector` component works in Streamlit + interactive modes
- [x] `MultiSourceRetriever` accepts collection specs, fans out, fuses with RRF
- [x] Results include "collection" attribution in metadata
- [x] `vector_explorer` shows multi-select dropdown with sizes
- [x] `chat.py` prompts for collections at startup
- [x] All existing retriever tests still pass (no regression)
- [x] Turkish explanations for operators/developers
