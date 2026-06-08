import sys
from pathlib import Path
import os
import re
import json
import glob
from datetime import datetime
import unicodedata

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import chromadb
from chromadb.config import Settings
from src.config.collections import COLLECTIONS
from src.config import settings
from src.common.span_resolver import resolve_spans_from_cache
from src.retriever.reranker import CrossEncoderReranker
import pandas as pd


@st.cache_resource
def _get_reranker():
    """Reranker modelini bir kez yükle, sonra cache'le."""
    return CrossEncoderReranker()


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


st.set_page_config(page_title="Tutanak Vector Explorer", layout="wide")
st.title("🏛️ Tutanak Vector DB Explorer")

# ─── Yardımcı: Tab2 metadata ────────────────────────────────────────────────


def _collection_prefix(spec) -> str:
    mapping = {
        "tbmm_minutes_docling_jina_v3": "d20",
        "tbmm_minutes_docling_jina_v4": "d20v4",
        "tbmm_tutanaklar_docling_jina_v3_4k": "tut4k",
        "tbmm_tutanaklar_nomic_v2": "tutnv2",
        "tbmm_tutanaklar_jina_v3_small": "tuts",
        settings.PRESS_COLLECTION: "gz",
        settings.MINUTES_COLLECTION: "min",
        settings.ONERGE_COLLECTION: "on",
    }
    return mapping.get(spec.name, spec.name.split("_")[-1][:6])


def _slugify(q: str) -> str:
    s = unicodedata.normalize("NFKD", q).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return "-".join(s.split("-")[:3])


def _next_id(target_path: str, prefix: str) -> str:
    try:
        data = json.loads(Path(target_path).read_text(encoding="utf-8"))
        used = [d.get("id", "") for d in data if d.get("id", "").startswith(prefix)]
        counter = len(used) + 1
    except Exception:
        counter = 1
    return f"{prefix}-{counter:03d}"


def _detect_years(resolved, collection) -> set[int]:
    if not resolved:
        return set()
    doc_ids = {s["document_id"] for s in resolved}
    try:
        got = collection.get(
            where={"document_id": {"$in": list(doc_ids)}},
            include=["metadatas"],
            limit=len(doc_ids) * 4,
        )
        years = set()
        for m in got["metadatas"] or []:
            d = m.get("document_date") or ""
            if len(d) >= 4 and d[:4].isdigit():
                years.add(int(d[:4]))
        return years
    except Exception:
        return set()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("Veritabanı Ayarları")

from src.ui.components.collection_selector import select_collections_streamlit
from src.retriever.multi_source import MultiSourceRetriever

selected_specs = select_collections_streamlit(
    defaults=["gazete_arsivi", "tbmm_minutes"]
)

if not selected_specs:
    st.error("En az bir koleksiyon seçiniz.")
    st.stop()

# Create multi-source retriever
retriever = MultiSourceRetriever(specs=selected_specs)

# Calculate total chunk count across selected collections
total_count = 0
for spec in selected_specs:
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=str(spec.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(spec.name)
        total_count += collection.count()
    except Exception:
        pass

st.sidebar.metric("Toplam Chunk Sayısı", total_count)

st.sidebar.divider()
st.sidebar.subheader("Reranker")
use_reranker = st.sidebar.checkbox(
    "🔀 Cross-Encoder Reranker Aktif",
    value=True,
    help="Aktif olduğunda arama sonuçları cross-encoder ile yeniden sıralanır.",
)

# Pending spans session state init
if "pending_spans" not in st.session_state:
    st.session_state["pending_spans"] = []

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Vektör Keşfi", "📝 Golden Data Oluşturucu", "📚 Mevcut Golden Data"])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Vektör Arama Testi")
    query = st.text_input("Arama yapmak için bir cümle yazın (Örn: Kardak kayalıkları)")
    top_k = st.slider("Sonuç Sayısı", 1, 20, 5)

    use_filter_extractor = st.checkbox(
        "⚙️ Auto Filter Extractor Aktif",
        value=True,
        help="Doğal dil sorgusundan metadata filtrelerini ve sadeleştirilmiş sorguyu otomatik çıkarır.",
    )

    pending_count = len(st.session_state["pending_spans"])
    if pending_count:
        st.info(f"📋 {pending_count} chunk Tab2 kuyruğunda bekliyor")

    if query:
        try:
            reranker = _get_reranker() if use_reranker else None

            # Filter Extraction
            search_query = query
            chroma_where = None

            fallback_level = None
            if use_filter_extractor:
                from src.generator.filter_extractor import FilterExtractor
                fe = FilterExtractor()
                extracted = fe.extract(query)
                refined = extracted.refined_query.strip()
                search_query = refined if refined else query
                chroma_where = fe.to_chroma_filter(extracted.filters)

                # Render dynamic visual feedback
                st.markdown("### ⚙️ Auto Filter Extractor Analiz Sonucu")
                col_ref, col_filt = st.columns(2)
                with col_ref:
                    st.info(f"🔍 **Sadeleştirilmiş Sorgu (Refined Query):** `{search_query}`")
                with col_filt:
                    if chroma_where:
                        st.success(f"🗂️ **Çıkarılan Metadata Filtreleri (Chroma DB Where):**\n```json\n{json.dumps(chroma_where, indent=2, ensure_ascii=False)}\n```")
                    else:
                        st.warning("ℹ️ Herhangi bir filtre çıkarılamadı (Sorgu doğrudan aratılıyor).")

            st.session_state["last_query"] = search_query

            with st.spinner("Aranıyor..."):
                per_collection = retriever.retrieve_per_collection(search_query, top_k=top_k)

            total_results = sum(len(r["documents"][0]) for r in per_collection.values())
            st.subheader(f"Sonuçlar ({total_results} döküman — {len(per_collection)} koleksiyon)")

            for coll_name, coll_result in per_collection.items():
                docs = coll_result["documents"][0]
                metas = coll_result["metadatas"][0]
                dists = coll_result["distances"][0]

                if not docs:
                    continue

                st.subheader(f"📊 {coll_name} ({len(docs)} sonuç)")

                for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                    score = 1 - dist
                    chunk_id = meta.get("document_id", "?")

                    with st.expander(f"Sonuç #{i+1} — {chunk_id} (Skor: {score:.4f})"):
                        st.write(doc)
                        st.code(chunk_id, language="text")
                        st.json(meta)

                        if st.button(f"➕ Golden Data'ya Ekle", key=f"add_{coll_name}_{chunk_id}_{i}"):
                            already = any(
                                p["chunk_id"] == chunk_id and p.get("collection") == coll_name
                                for p in st.session_state["pending_spans"]
                            )
                            if already:
                                st.warning("Zaten kuyruğa eklendi.")
                            else:
                                st.session_state["pending_spans"].append({
                                    "chunk_id": chunk_id,
                                    "collection": coll_name,
                                    "text_preview": doc[:200],
                                    "text_full": doc,
                                })
                                st.success(f"Kuyruğa eklendi: {chunk_id}")
                                st.rerun()

        except Exception as e:
            st.error(f"Arama hatası: {e}")

    st.header("📄 Tüm Veriler (İlk 50 — Seçili Koleksiyonlardan)")

    df_data = []
    for spec in selected_specs:
        try:
            client = chromadb.PersistentClient(
                path=str(spec.db_path),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(spec.name)
            all_data = collection.get(limit=50, include=["documents", "metadatas"])

            for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                df_data.append({
                    "Koleksiyon": spec.name,
                    "ID": cid,
                    "Metin": doc[:100] + "...",
                    "Tarih": meta.get("document_date", "-"),
                    "Birleşim": meta.get("session", "-"),
                })
        except Exception as e:
            st.warning(f"Koleksiyon '{spec.name}' okunamadı: {e}")

    if df_data:
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
    else:
        st.warning("Seçili koleksiyonlar boş.")

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("✨ Yeni Değerlendirme Sorusu Ekle")

    fixture_dir = root_path / "tests" / "fixtures"
    fixture_files = sorted(glob.glob(str(fixture_dir / "eval_*.json")))
    file_options = {os.path.basename(f): f for f in fixture_files}

    if not file_options:
        st.warning(f"Fixtures klasöründe ({fixture_dir}) JSON dosyası bulunamadı.")
    else:
        selected_file = st.selectbox("Eklenecek JSON Dosyası", list(file_options.keys()), key="save_file_select")
        target_path = file_options[selected_file]

        # ─── Auto-resolve on queue change ───
        pending = st.session_state["pending_spans"]
        sig = tuple(p["chunk_id"] for p in pending)
        if sig != st.session_state.get("_pending_signature"):
            if pending:
                # For multi-source: resolve from first spec that has the chunk
                # This is a workaround; ideally resolve_spans_from_cache should handle multi-source
                resolved = []
                errors = []
                for p in pending:
                    found = False
                    for spec in selected_specs:
                        try:
                            span_resolved, span_errors = resolve_spans_from_cache([p["chunk_id"]], spec)
                            if span_resolved:
                                resolved.extend(span_resolved)
                                found = True
                                break
                        except Exception:
                            continue
                    if not found:
                        errors.append(f"Chunk bulunamadı: {p['chunk_id']}")
                st.session_state["resolved_spans"] = resolved
                st.session_state["_resolve_errors"] = errors
            else:
                st.session_state["resolved_spans"] = []
                st.session_state["_resolve_errors"] = []
            st.session_state["_pending_signature"] = sig

        resolved = st.session_state.get("resolved_spans", [])
        errors = st.session_state.get("_resolve_errors", [])

        # ─── Queue display ───
        if pending:
            st.subheader(f"📋 Kuyruk ({len(pending)} chunk, {len(resolved)} başarılı)")
            for i, p in enumerate(pending):
                resolved_item = next((r for r in resolved if r.get("_chunk_id") == p["chunk_id"]), None)
                if resolved_item:
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        st.markdown(
                            f"**`{p['chunk_id']}`** → chars **{resolved_item['char_start']}–{resolved_item['char_end']}**"
                        )
                        with st.expander("Tam metni göster"):
                            st.write(resolved_item.get("_text", resolved_item.get("_preview", "")))
                    with col_b:
                        if st.button("🗑️", key=f"remove_{i}"):
                            st.session_state["pending_spans"].pop(i)
                            st.rerun()

            if errors:
                with st.expander("⚠️ Hata Detayları"):
                    for e in errors:
                        st.warning(e)
        else:
            st.info("Tab1'de arama yapıp ➕ butonu ile chunk ekleyin.")

        st.divider()

        # ─── Form ───
        if not resolved:
            st.error("Soru kaydetmek için önce kuyrukta chunk olmalı.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                default_query = st.session_state.get("last_query", "")
                q_query = st.text_area(
                    "Soru Metni",
                    value=default_query,
                    help="Tab1'deki son aramadan otomatik dolduruldu. İsterseniz düzenleyin.",
                    key="tab2_query_input",
                )

            # Compute auto-filled golden answer from resolved spans
            auto_golden = "\n\n---\n\n".join(
                s.get("_text", "") for s in resolved if s.get("_text")
            ).strip()

            with col2:
                if not st.session_state.get("tab2_golden_override"):
                    st.session_state["tab2_golden_input"] = auto_golden

                q_golden_answer = st.text_area(
                    "Golden Answer (otomatik — düzenleyebilirsiniz)",
                    value=st.session_state.get("tab2_golden_input", auto_golden),
                    key="tab2_golden_input",
                    height=200,
                )

                if st.session_state.get("tab2_golden_input") != auto_golden:
                    st.session_state["tab2_golden_override"] = True

            # Auto-fill ID — use prefix from first selected collection
            first_spec = selected_specs[0] if selected_specs else None
            prefix = _collection_prefix(first_spec) if first_spec else "id"
            slug = _slugify(q_query) if q_query else "yeni-soru"
            auto_id = f"{prefix}-{_next_id(target_path, f'{prefix}-{slug}')}"

            if st.session_state.get("tab2_id_override"):
                q_id = st.text_input(
                    "Soru ID'si",
                    value=st.session_state.get("tab2_id", auto_id),
                    key="tab2_id_input",
                )
            else:
                q_id = st.text_input(
                    "Soru ID'si",
                    value=auto_id,
                    key="tab2_id_input",
                )

            if st.session_state.get("tab2_id_input") != auto_id:
                st.session_state["tab2_id_override"] = True

            # Auto-fill year — detect from all selected collections
            years = set()
            for spec in selected_specs:
                try:
                    client = chromadb.PersistentClient(
                        path=str(spec.db_path),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    collection = client.get_collection(spec.name)
                    years.update(_detect_years(resolved, collection))
                except Exception:
                    continue
            year_auto = next(iter(years)) if len(years) == 1 else None

            if len(years) == 1 and not st.session_state.get("tab2_year_override"):
                q_expected_year = st.number_input(
                    "Beklenen Yıl",
                    value=year_auto,
                    key="tab2_year_input",
                )
            elif len(years) > 1:
                st.warning(f"Birden fazla yıl tespit edildi: {sorted(years)}. Manuel seçin.")
                q_expected_year = st.number_input(
                    "Beklenen Yıl",
                    value=None,
                    key="tab2_year_input",
                )
            else:
                q_expected_year = st.number_input(
                    "Beklenen Yıl",
                    value=None,
                    key="tab2_year_input",
                )

            if st.session_state.get("tab2_year_input") != year_auto:
                st.session_state["tab2_year_override"] = True

            # Tags with chip multiselect
            preset_tags = ["topic", "date-filter", "source-routing", "gazete-only", "minutes-only", "custom"]
            extras = st.session_state.setdefault("tab2_tag_extras", [])
            q_tags = st.multiselect(
                "Etiketler",
                options=preset_tags + extras,
                default=st.session_state.get("tab2_tags", []),
                key="tab2_tags_select",
            )
            new_tag = st.text_input("Yeni etiket ekle", key="tab2_new_tag")
            if new_tag and new_tag not in preset_tags + extras:
                extras.append(new_tag)
                st.rerun()

            # ─── Advanced Settings Expander ───
            with st.expander("🛠️ Gelişmiş Ayarlar"):
                INTENT_OPTIONS = ["factual", "list", "comparison", "multi-hop", "no-answer"]
                q_intent = st.selectbox(
                    "Intent",
                    options=[""] + INTENT_OPTIONS,
                    index=0,
                    help=(
                        "factual: tek atomik gerçek soru\n"
                        "list: liste/enumerasyon istenir\n"
                        "comparison: karşılaştırma\n"
                        "multi-hop: birden fazla kanıt parçası gerekir\n"
                        "no-answer: cevap koleksiyon dışında, model abstain etmeli"
                    ),
                    key="tab2_intent_input",
                )
                q_abstain = st.checkbox(
                    "Abstain required (model 'bilmiyorum' demeli)",
                    value=False,
                    help="Soru bu koleksiyonda yanıtlanamaz; doğru cevap abstain'dir.",
                    key="tab2_abstain_input",
                )

            # Save button
            if st.button("💾 Kaydet", type="primary", use_container_width=True):
                if not q_query:
                    st.error("Soru Metni zorunlu!")
                elif not resolved:
                    st.error("Span bulunamadı.")
                else:
                    clean_spans = [
                        {k: v for k, v in s.items() if not k.startswith("_")}
                        for s in resolved
                    ]
                    new_entry = {
                        "id": q_id,
                        "query": q_query,
                        "intent": q_intent or None,
                        "expected_year": int(q_expected_year) if q_expected_year else None,
                        "gold_evidence_spans": clean_spans,
                        "golden_answer": q_golden_answer.strip() or None,
                        "abstain_required": bool(q_abstain),
                        "tags": q_tags,
                    }

                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data.append(new_entry)
                        with open(target_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        st.success(f"Eklendi! Toplam: {len(data)} soru")
                        st.json(new_entry)

                        # Clear queue and form state
                        st.session_state["pending_spans"] = []
                        st.session_state["resolved_spans"] = []
                        st.session_state["_pending_signature"] = ()
                        st.session_state["_resolve_errors"] = []
                        st.session_state["tab2_tags"] = []
                        st.session_state["tab2_id_override"] = False
                        st.session_state["tab2_year_override"] = False
                        st.session_state["tab2_golden_override"] = False
                        st.session_state["tab2_new_tag"] = ""
                        st.session_state.pop("last_query", None)

                        st.rerun()

                    except Exception as e:
                        st.error(f"Kayıt hatası: {e}")

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("📚 Mevcut Golden Data Soruları")

    fixture_dir = root_path / "tests" / "fixtures"
    fixture_files = sorted(glob.glob(str(fixture_dir / "eval_*.json")))
    file_options = {os.path.basename(f): f for f in fixture_files}

    if file_options:
        view_selected_file = st.selectbox(
            "İncelenecek JSON Dosyası", list(file_options.keys()), key="view_file_select"
        )
        view_target_path = file_options[view_selected_file]

        try:
            with open(view_target_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            st.metric("Toplam Soru Sayısı", len(existing_data))

            # Span vs chunk_id istatistiği
            span_count = sum(1 for d in existing_data if "gold_evidence_spans" in d or "relevant_spans" in d)
            chunk_count = sum(1 for d in existing_data if "relevant_chunk_ids" in d)
            c1, c2 = st.columns(2)
            c1.metric("Span tabanlı", span_count)
            c2.metric("Chunk ID tabanlı (eski)", chunk_count)

            search_query = st.text_input("Ara", "").lower()

            filtered_data = []
            for item in existing_data:
                if search_query and search_query not in item["query"].lower() and not any(
                    search_query in t.lower() for t in item.get("tags", [])
                ):
                    continue
                filtered_data.append({
                    "ID": item["id"],
                    "Soru": item["query"],
                    "Yıl": item.get("expected_year", "-"),
                    "Etiketler": ", ".join(item.get("tags", [])),
                    "Format": (
                        "evidence_spans" if "gold_evidence_spans" in item
                        else "span" if "relevant_spans" in item
                        else "chunk_id"
                    ),
                    "Span/Chunk": len(
                        item.get("gold_evidence_spans")
                        or item.get("relevant_spans")
                        or item.get("relevant_chunk_ids", [])
                    ),
                })

            if filtered_data:
                st.dataframe(pd.DataFrame(filtered_data), use_container_width=True)
            else:
                st.info("Kriter uyuşmadı.")

        except Exception as e:
            st.error(f"Okuma hatası: {e}")
