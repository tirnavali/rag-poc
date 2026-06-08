"""Tests for AuthorSegmentExtractor + tag_atoms state machine."""
from __future__ import annotations

import pytest

from src.common.parsing.author_extractor import (
    AuthorSegmentExtractor,
    AuthorTransition,
    TaggedAtom,
    tag_atoms,
    tag_chunks_post_hoc,
)
from src.common.parsing.extractors import get_extractor
from src.common.parsing.extractors.gazete import GazeteAuthorExtractor
from src.common.parsing.extractors.noop import NoopAuthorExtractor
from src.common.parsing.extractors.onerge import OnergeAuthorExtractor
from src.common.parsing.extractors.tutanak import TutanakAuthorExtractor


def _atom(text: str, label: str = "Text") -> dict:
    return {"text": text, "label": label}


# ─── TutanakAuthorExtractor ──────────────────────────────────────


class TestTutanakExtractor:
    def setup_method(self):
        self.ex = TutanakAuthorExtractor()

    def test_chair_pattern(self):
        t = self.ex.detect_transition("BAŞKAN — Sayın milletvekilleri, oturumu açıyorum.")
        assert t is not None
        assert t.author == "BAŞKAN"
        assert t.author_role == "başkan"

    def test_chair_vekili(self):
        t = self.ex.detect_transition("BAŞKANVEKİLİ — Buyurun.")
        assert t is not None
        assert t.author == "BAŞKANVEKİLİ"

    def test_deputy_with_constituency(self):
        t = self.ex.detect_transition("AHMET YILMAZ (İstanbul) — Teşekkür ederim.")
        assert t is not None
        assert t.author == "AHMET YILMAZ"
        assert t.author_role == "milletvekili"
        assert t.extra["constituency"] == "İstanbul"

    def test_minister(self):
        t = self.ex.detect_transition("DEVLET BAKANI HASAN GEMİCİ — Söz alıyorum.")
        assert t is not None
        assert "GEMİCİ" in t.author

    def test_no_match(self):
        assert self.ex.detect_transition("Sayın Başkan, izninizle...") is None

    def test_continuation_text(self):
        assert self.ex.detect_transition("Bu konuya değinmek istiyorum.") is None

    def test_compound_minister_title(self):
        t = self.ex.detect_transition(
            "DIŞİŞLERİ BAKANI VE BAŞBAKAN YARDIMCISI DENİZ BAYKAL — Konuşuyorum."
        )
        assert t is not None
        assert t.author == "DENİZ BAYKAL"
        assert "dişi̇şleri̇ bakani ve başbakan yardimcisi" in t.author_role or "dişişleri bakani ve başbakan yardimcisi" in t.author_role

    def test_double_space_normalization(self):
        t1 = self.ex.detect_transition(
            "DEVLET BAKANI  HASAN GEMİCİ — Söz alıyorum."
        )
        t2 = self.ex.detect_transition("DEVLET BAKANI HASAN GEMİCİ — Söz alıyorum.")
        assert t1 is not None and t2 is not None
        assert t1.author == t2.author
        assert t1.author_role == t2.author_role

    def test_confidence_heuristics_long_name_with_title(self):
        t = self.ex.detect_transition(
            "DEVLET BAKANI VERY LONG NAME WITH MANY WORDS THAT IS OVER LIMIT — Söz."
        )
        assert t is not None
        assert t.confidence < 1.0

    def test_confidence_high_on_clean(self):
        t = self.ex.detect_transition("AHMET YILMAZ (İstanbul) — Söz.")
        assert t is not None
        assert t.confidence == 1.0


# ─── GazeteAuthorExtractor ────────────────────────────────────────


class TestGazeteExtractor:
    def setup_method(self):
        self.ex = GazeteAuthorExtractor()

    def test_yazan_pattern(self):
        t = self.ex.detect_transition("Yazan: Ahmet Hakan")
        assert t is not None
        assert t.author == "Ahmet Hakan"
        assert t.author_role == "köşe yazarı"

    def test_byline_pattern(self):
        t = self.ex.detect_transition("Mehmet Yılmaz — Hürriyet")
        assert t is not None
        assert t.author == "Mehmet Yılmaz"

    def test_no_match(self):
        assert self.ex.detect_transition("Bugün yeni bir gelişme yaşandı.") is None


# ─── OnergeAuthorExtractor ────────────────────────────────────────


class TestOnergeExtractor:
    def setup_method(self):
        self.ex = OnergeAuthorExtractor()

    def test_proposer_with_co_signers(self):
        text = "Konya Milletvekili Ahmet Yılmaz ve 14 arkadaşının kanun teklifi"
        t = self.ex.detect_transition(text)
        assert t is not None
        assert t.author == "Ahmet Yılmaz"
        assert t.extra["constituency"] == "Konya"
        assert t.extra["co_signers_count"] == 14

    def test_proposer_solo(self):
        text = "İstanbul Milletvekili Mehmet Kaya tarafından sunulan teklif"
        t = self.ex.detect_transition(text)
        assert t is not None
        assert t.author == "Mehmet Kaya"

    def test_signature_list(self):
        text = "(İmza: AHMET YILMAZ, MEHMET KAYA, FATMA ÖZ)"
        t = self.ex.detect_transition(text)
        assert t is not None
        assert t.extra["signatories"] == ["AHMET YILMAZ", "MEHMET KAYA", "FATMA ÖZ"]


# ─── NoopAuthorExtractor ──────────────────────────────────────────


class TestNoopExtractor:
    def test_never_detects(self):
        ex = NoopAuthorExtractor()
        assert ex.detect_transition("BAŞKAN — açıklama") is None
        assert ex.detect_transition("Yazan: Foo") is None


# ─── Registry ────────────────────────────────────────────────────


class TestRegistry:
    def test_known_types(self):
        assert isinstance(get_extractor("tutanak"), TutanakAuthorExtractor)
        assert isinstance(get_extractor("press_clip"), GazeteAuthorExtractor)
        assert isinstance(get_extractor("onerge"), OnergeAuthorExtractor)
        assert isinstance(get_extractor("kanun_teklifi"), OnergeAuthorExtractor)

    def test_unknown_falls_back_to_noop(self):
        ex = get_extractor("unknown_type")
        assert isinstance(ex, NoopAuthorExtractor)


# ─── tag_atoms (state machine) ─────────────────────────────────


class TestTagAtoms:
    def test_state_propagation(self):
        atoms = [
            _atom("AHMET YILMAZ (İstanbul) — Söz alıyorum."),
            _atom("Bu konuda görüşlerimi belirtmek istiyorum."),
            _atom("Birinci konu şudur..."),
        ]
        tagged = tag_atoms(atoms, TutanakAuthorExtractor())
        assert all(t.author == "AHMET YILMAZ" for t in tagged)
        assert tagged[0].is_continuation is False
        assert tagged[1].is_continuation is True
        assert tagged[2].is_continuation is True
        # All atoms in same segment
        assert tagged[0].segment_index == tagged[1].segment_index == tagged[2].segment_index

    def test_segment_index_increments(self):
        atoms = [
            _atom("BAŞKAN — Açıyorum."),
            _atom("Açıklama yapıyorum."),
            _atom("AHMET YILMAZ (İzmir) — Teşekkürler."),
            _atom("Konuşmama başlıyorum."),
        ]
        tagged = tag_atoms(atoms, TutanakAuthorExtractor())
        assert tagged[0].author == "BAŞKAN"
        assert tagged[1].author == "BAŞKAN"
        assert tagged[2].author == "AHMET YILMAZ"
        assert tagged[3].author == "AHMET YILMAZ"
        assert tagged[2].segment_index == tagged[0].segment_index + 1

    def test_initial_author_fallback(self):
        atoms = [_atom("Hiçbir konuşmacı pattern yok."), _atom("Devam metni.")]
        tagged = tag_atoms(
            atoms,
            NoopAuthorExtractor(),
            initial_author="Fallback Kişi",
            initial_role="başkan",
        )
        assert all(t.author == "Fallback Kişi" for t in tagged)
        assert all(t.author_role == "başkan" for t in tagged)
        assert all(t.is_continuation for t in tagged)

    def test_empty_atoms(self):
        assert tag_atoms([], TutanakAuthorExtractor()) == []

    def test_extra_carried_per_atom(self):
        atoms = [
            _atom("AHMET YILMAZ (Bursa) — Söz."),
            _atom("Devam ediyorum."),
        ]
        tagged = tag_atoms(atoms, TutanakAuthorExtractor())
        assert tagged[0].extra["constituency"] == "Bursa"
        assert tagged[1].extra["constituency"] == "Bursa"


# ─── tag_chunks_post_hoc (HybridChunker path) ─────────────────


class TestTagChunksPostHoc:
    def test_state_propagates_across_chunks(self):
        chunks = [
            {"text": "AHMET YILMAZ (İzmir) — Söz alıyorum.\n\nKonu üzerinde duruyorum."},
            {"text": "Devam ediyorum, ikinci chunk."},
            {"text": "BAŞKAN — Teşekkürler.\n\nSayın milletvekilleri..."},
        ]
        tag_chunks_post_hoc(chunks, TutanakAuthorExtractor())
        assert chunks[0]["metadata"]["author"] == "AHMET YILMAZ"
        assert chunks[0]["metadata"]["starts_mid_segment"] is False
        # State carried into chunk 2
        assert chunks[1]["metadata"]["author"] == "AHMET YILMAZ"
        assert chunks[1]["metadata"]["starts_mid_segment"] is True
        # Chunk 3 detects new author
        assert chunks[2]["metadata"]["author"] == "BAŞKAN"
        assert chunks[2]["metadata"]["starts_mid_segment"] is False

    def test_initial_author_fallback_when_no_transition(self):
        chunks = [{"text": "Sıradan rapor metni, hiç konuşmacı pattern yok."}]
        tag_chunks_post_hoc(
            chunks,
            NoopAuthorExtractor(),
            initial_author="Komisyon",
            initial_role="kurul",
        )
        assert chunks[0]["metadata"]["author"] == "Komisyon"
        assert chunks[0]["metadata"]["author_role"] == "kurul"
        assert chunks[0]["metadata"]["starts_mid_segment"] is True

    def test_multiple_authors_in_single_chunk(self):
        chunks = [
            {
                "text": (
                    "AHMET YILMAZ (Adana) — Birinci konuşma.\n\n"
                    "MEHMET KAYA (Ankara) — İkinci konuşma."
                )
            }
        ]
        tag_chunks_post_hoc(chunks, TutanakAuthorExtractor())
        meta = chunks[0]["metadata"]
        assert set(meta["authors_in_chunk"]) == {"AHMET YILMAZ", "MEHMET KAYA"}
        assert len(meta["segment_indices"]) == 2

    def test_empty_chunks(self):
        assert tag_chunks_post_hoc([], TutanakAuthorExtractor()) == []
