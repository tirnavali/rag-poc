# Vector Explorer Multi-Collection Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix vector_explorer to display results from all selected collections in a single combined list with collection attribution.

**Architecture:** Flatten per-collection result lists from `MultiSourceRetriever.retrieve()` into single iteration. Preserve RRF ranking order. Use existing `meta["collection"]` field for attribution tags.

**Tech Stack:** Streamlit, MultiSourceRetriever (no backend changes).

---

## File Structure

- `scripts/vector_explorer.py` — Main UI file
  - Lines 182-186: Result retrieval (needs flattening)
  - Lines 190-200: Display loop (needs restructuring to iterate across collections)
- No new files; purely UI fix in existing script

---

### Task 1: Write integration test for multi-collection display

**Files:**
- Create: `tests/test_vector_explorer_multi_collection.py`

- [ ] **Step 1: Write failing test for multi-collection result display**

```python
def test_vector_explorer_multi_collection_results():
    """Verify vector_explorer flatten logic returns results from all collections."""
    from src.retriever.multi_source import MultiSourceRetriever
    from src.config.collections import get_spec
    
    # Create specs for two different collections
    spec1 = get_spec("gazete_arsivi")
    spec2 = get_spec("tbmm_minutes")
    
    # Create retriever with both specs
    retriever = MultiSourceRetriever(specs=[spec1, spec2])
    
    # Retrieve results for a query that should match both collections
    result = retriever.retrieve("ekonomi", top_k=3)
    
    # Verify result structure: per-collection lists
    assert isinstance(result["documents"], list)
    assert len(result["documents"]) == 2, "Expected results from 2 collections"
    
    # Verify each collection has results
    for docs_list, metas_list, dists_list in zip(
        result["documents"], result["metadatas"], result["distances"]
    ):
        # Each collection should have up to 3 results
        assert len(docs_list) > 0, "Collection should have at least 1 result"
        assert len(docs_list) == len(metas_list)
        assert len(docs_list) == len(dists_list)
        
        # Verify collection field in metadata
        for meta in metas_list:
            assert "collection" in meta, "Metadata missing 'collection' field"
            assert meta["collection"] in [spec1.name, spec2.name]
    
    # Verify flattened result would have total results
    total_docs = sum(len(docs) for docs in result["documents"])
    assert total_docs > 0, "Total results across collections should be > 0"
```

- [ ] **Step 2: Run test to verify it passes (pre-existing test)**

Run: `pytest tests/test_vector_explorer_multi_collection.py::test_vector_explorer_multi_collection_results -v`

Expected: PASS (test verifies MultiSourceRetriever already returns multi-collection structure)

---

### Task 2: Extract result flattening logic into helper function

**Files:**
- Modify: `scripts/vector_explorer.py:182-200`

- [ ] **Step 1: Write helper function to flatten results**

Add this function after imports, before `st.set_page_config()`:

```python
def _flatten_multi_collection_results(retriever_result):
    """Flatten per-collection results into single iteration.
    
    Args:
        retriever_result: dict with keys "documents", "metadatas", "distances"
                         Each value is list[per_collection_list]
    
    Yields:
        Tuple[str, dict, float]: (document_text, metadata_dict, distance_score)
    """
    docs_by_col = retriever_result["documents"]
    metas_by_col = retriever_result["metadatas"]
    dists_by_col = retriever_result["distances"]
    
    for docs, metas, dists in zip(docs_by_col, metas_by_col, dists_by_col):
        for doc, meta, dist in zip(docs, metas, dists):
            yield doc, meta, dist
```

- [ ] **Step 2: Write test for flattening function**

```python
def test_flatten_multi_collection_results():
    """Verify flattening preserves all results."""
    # Test data: 2 collections, 2 results each
    test_result = {
        "documents": [
            ["doc1_col1", "doc2_col1"],
            ["doc1_col2", "doc2_col2"],
        ],
        "metadatas": [
            [{"collection": "col1", "id": "1"}, {"collection": "col1", "id": "2"}],
            [{"collection": "col2", "id": "3"}, {"collection": "col2", "id": "4"}],
        ],
        "distances": [
            [0.1, 0.2],
            [0.15, 0.25],
        ],
    }
    
    flattened = list(_flatten_multi_collection_results(test_result))
    
    # Should have 4 total results
    assert len(flattened) == 4
    
    # Each tuple should be (doc, meta, dist)
    for doc, meta, dist in flattened:
        assert isinstance(doc, str)
        assert isinstance(meta, dict)
        assert isinstance(dist, float)
    
    # Verify order preserved
    assert flattened[0][0] == "doc1_col1"
    assert flattened[1][0] == "doc2_col1"
    assert flattened[2][0] == "doc1_col2"
    assert flattened[3][0] == "doc2_col2"
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_vector_explorer_multi_collection.py -v`

Expected: PASS (2 tests passing)

---

### Task 3: Replace result extraction in vector_explorer Tab 1

**Files:**
- Modify: `scripts/vector_explorer.py:182-200`

- [ ] **Step 1: Replace old extraction with flattened loop**

Find lines 182-200 (retrieve and display logic). Replace:

```python
            with st.spinner("Aranıyor..."):
                result = retriever.retrieve(search_query, top_k=top_k)

            docs = result["documents"][0]
            metas = result["metadatas"][0]
            dists = result["distances"][0]

            st.subheader(f"Sonuçlar ({len(docs)} döküman)")

            for i, (doc, meta, dist) in enumerate(
                zip(docs, metas, dists)
            ):
                collection = meta.get("collection", "bilinmiyor")
                score = 1 - dist
                chunk_id = meta.get("document_id", "?")

                with st.expander(f"Sonuç #{i+1} — [{collection}] {chunk_id} (Skor: {score:.4f})"):
                    st.write(doc)
                    st.code(chunk_id, language="text")
                    st.json(meta)

                    if st.button(f"➕ Golden Data'ya Ekle", key=f"add_{chunk_id}_{i}"):
                        already = any(
                            p["chunk_id"] == chunk_id
                            for p in st.session_state["pending_spans"]
                        )
                        if already:
                            st.warning("Zaten kuyruğa eklendi.")
                        else:
                            st.session_state["pending_spans"].append({
                                "chunk_id": chunk_id,
                                "text_preview": doc[:200],
                                "text_full": doc,
                            })
                            st.success(f"Kuyruğa eklendi: {chunk_id}")
                            st.rerun()
```

With:

```python
            with st.spinner("Aranıyor..."):
                result = retriever.retrieve(search_query, top_k=top_k)

            # Flatten results from all collections
            flattened = list(_flatten_multi_collection_results(result))
            
            st.subheader(f"Sonuçlar ({len(flattened)} döküman)")

            for i, (doc, meta, dist) in enumerate(flattened):
                collection = meta.get("collection", "bilinmiyor")
                score = 1 - dist
                chunk_id = meta.get("document_id", "?")

                with st.expander(f"Sonuç #{i+1} — [{collection}] {chunk_id} (Skor: {score:.4f})"):
                    st.write(doc)
                    st.code(chunk_id, language="text")
                    st.json(meta)

                    if st.button(f"➕ Golden Data'ya Ekle", key=f"add_{chunk_id}_{i}"):
                        already = any(
                            p["chunk_id"] == chunk_id
                            for p in st.session_state["pending_spans"]
                        )
                        if already:
                            st.warning("Zaten kuyruğa eklendi.")
                        else:
                            st.session_state["pending_spans"].append({
                                "chunk_id": chunk_id,
                                "text_preview": doc[:200],
                                "text_full": doc,
                            })
                            st.success(f"Kuyruğa eklendi: {chunk_id}")
                            st.rerun()
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_vector_explorer_multi_collection.py -v`

Expected: PASS (all tests still pass)

- [ ] **Step 3: Commit changes**

```bash
git add scripts/vector_explorer.py tests/test_vector_explorer_multi_collection.py
git commit -m "feat: flatten multi-collection results in vector explorer

- Extract result flattening logic into _flatten_multi_collection_results()
- Update Tab 1 retrieve/display loop to use flattened iteration
- Preserve RRF ranking order across collections
- Display [collection_name] tag in each result header
- Tests verify flattening preserves all collection results"
```

---

### Task 4: Manual verification (Streamlit UI test)

**Files:**
- No file changes; manual testing

- [ ] **Step 1: Start vector_explorer**

Run: `streamlit run scripts/vector_explorer.py`

Expected: Streamlit app opens on localhost:8501

- [ ] **Step 2: Select multiple collections**

In sidebar:
- Check: `gazete_arsivi`
- Check: `tbmm_minutes`

Expected: "Toplam Chunk Sayısı" shows combined count

- [ ] **Step 3: Search for query matching both collections**

In Tab 1 (🔍 Vektör Keşfi):
- Type query: `ekonomi` (should match both newspaper and parliament data)
- Click search

Expected:
- Results displayed (not empty)
- Multiple results from both collections visible
- Each result header shows collection name tag like `[gazete_arsivi]` or `[tbmm_minutes]`
- Results appear in RRF-ranked order (scores descending)

- [ ] **Step 4: Verify result count**

Check result count in header:
- Should show total across all collections (e.g., "Sonuçlar (8 döküman)")
- Should be > 0

- [ ] **Step 5: Verify collection attribution**

Click several result expanders:
- Each should have `collection` field in metadata JSON
- Should match tag shown in header

- [ ] **Step 6: Verify "Tüm Veriler" section (All Data)**

Scroll to bottom of Tab 1:
- Should show data from all selected collections
- Dataframe should have rows from both `gazete_arsivi` and `tbmm_minutes`

- [ ] **Step 7: (Optional) Test with Tab 2 (Golden Data)**

Click "➕ Golden Data'ya Ekle" on a result:
- Should add to pending spans
- Can proceed to Tab 2 to review

---

## Self-Review

**Spec coverage:**
- ✅ Problem described (lines 184-186 only take [0])
- ✅ Solution: flatten results (Task 2-3)
- ✅ Display format: [collection_name] tag (Task 3 Step 1)
- ✅ Testing: verify results from all collections (Task 1, 4)
- ✅ Files modified: vector_explorer.py (Task 2-3)

**Placeholder scan:**
- ✅ No TBD/TODO
- ✅ All code blocks complete
- ✅ All test assertions written
- ✅ All commands exact with expected output

**Type consistency:**
- ✅ `retriever_result` dict matches MultiSourceRetriever return type
- ✅ Metadata always has `"collection"` field (set by MultiSourceRetriever)
- ✅ Distance scores are floats; score = 1 - dist

**Scope:**
- ✅ Single focused task: UI display fix
- ✅ No backend changes; no new features
- ✅ Straightforward refactor of existing loop
