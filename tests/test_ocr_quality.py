"""Tier-1 OCR kalite kontrolü testleri (Docling mock'lanır, offline çalışır).

Kapsam:
- compute_quality bayrak kuralları (atom yoğunluğu, karakter sapması, OCR güveni)
- extract_ocr_confidence'ın Docling sürümleri arasında savunmacı davranışı
- turkish_ocr_signals sezgisel sayaçları
- MarkdownConverter: quality alanı olmayan eski Level-1 cache'in geçerli sayılması
- DoclingManager.pack: ocr_flagged'in tüm yollardan chunk metadata'sına taşınması
- DocumentManifest: opsiyonel quality_json alanı
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.parsing import quality as q
from src.common.parsing.docling_manager import DoclingManager
from src.common.parsing.markdown_converter import MarkdownConverter, ParsedDocument
from src.config import settings
from src.trainer.ingestion.adapters.base import DocumentInput
from src.trainer.ingestion.manifest import DocumentManifest


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Cache/artefakt/istatistik yollarını tmp'e yönlendirir."""
    parse_cache = tmp_path / "parse_cache"
    parse_cache.mkdir(parents=True)
    monkeypatch.setattr(settings, "PARSE_CACHE_DIR", parse_cache)
    monkeypatch.setattr(settings, "MARKDOWN_DIR", tmp_path / "markdown")
    monkeypatch.setattr(settings, "PAGES_DIR", tmp_path / "pages")
    monkeypatch.setattr(settings, "QUALITY_STATS_FILE", parse_cache / "quality_stats.json")
    return tmp_path


def _atoms(count: int, page: int = 1):
    return [
        {"text": f"Atom {i}", "label": "text", "page": page, "pages": [page]}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# compute_quality — bayrak kuralları
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_low_atom_density_flag(self, isolated_dirs):
        # 2 atom / 10 sayfa → 0.2 < QUALITY_MIN_ATOMS_PER_PAGE
        atoms = [
            {"text": "a", "label": "text", "page": 1, "pages": [1]},
            {"text": "b", "label": "text", "page": 10, "pages": [10]},
        ]
        result = q.compute_quality(atoms, "a\n\nb")
        assert "low_atom_density" in result["flags"]
        assert result["ocr_flagged"] is True
        assert result["page_count"] == 10
        assert result["empty_page_count"] == 8

    def test_dense_document_no_flags(self, isolated_dirs):
        result = q.compute_quality(_atoms(10, page=1), "x" * 500)
        assert result["flags"] == []
        assert result["ocr_flagged"] is False

    def test_no_page_metadata_skips_density_check(self, isolated_dirs):
        atoms = [{"text": "a", "label": "text", "page": None, "pages": []}]
        result = q.compute_quality(atoms, "a")
        assert result["atoms_per_page"] is None
        assert "low_atom_density" not in result["flags"]

    def test_char_count_outlier_flag(self, isolated_dirs):
        # 3 belge ~1000 karakter/sayfa ile istatistik tohumla
        for i in range(3):
            result = q.compute_quality(
                _atoms(10), "x" * 1000, document_type="tutanak", stats_key=f"doc{i}"
            )
            assert "char_count_outlier" not in result["flags"]

        # 4. belge: 500 karakter/sayfa → %50 sapma > %30 → bayrak
        result = q.compute_quality(
            _atoms(10), "x" * 500, document_type="tutanak", stats_key="outlier"
        )
        assert "char_count_outlier" in result["flags"]
        assert result["ocr_flagged"] is True

    def test_char_stats_idempotent_reingest(self, isolated_dirs):
        for i in range(3):
            q.compute_quality(
                _atoms(10), "x" * 1000, document_type="tutanak", stats_key=f"doc{i}"
            )
        # Aynı stats_key ile tekrar — kendi değeri ortalamadan hariç tutulur,
        # bayrak kararlı kalır
        for _ in range(2):
            result = q.compute_quality(
                _atoms(10), "x" * 500, document_type="tutanak", stats_key="outlier"
            )
            assert "char_count_outlier" in result["flags"]

    def test_char_check_skipped_without_document_type(self, isolated_dirs):
        result = q.compute_quality(_atoms(10), "x" * 5, stats_key="h")
        assert "char_count_outlier" not in result["flags"]
        assert not settings.QUALITY_STATS_FILE.exists()

    def test_low_ocr_confidence_flag(self, isolated_dirs):
        flagged = q.compute_quality(_atoms(10), "x" * 100, ocr_mean_confidence=0.5)
        assert "low_ocr_confidence" in flagged["flags"]

        clean = q.compute_quality(_atoms(10), "x" * 100, ocr_mean_confidence=0.95)
        assert "low_ocr_confidence" not in clean["flags"]

        # None → kontrol atlanır (eski cache / OCR kapalı)
        unknown = q.compute_quality(_atoms(10), "x" * 100, ocr_mean_confidence=None)
        assert "low_ocr_confidence" not in unknown["flags"]
        assert unknown["ocr_mean_confidence"] is None


# ---------------------------------------------------------------------------
# extract_ocr_confidence — Docling erişim yolları
# ---------------------------------------------------------------------------

class TestExtractOcrConfidence:
    def test_from_confidence_report(self):
        result = SimpleNamespace(confidence=SimpleNamespace(ocr_score=0.91), pages=[])
        assert q.extract_ocr_confidence(result) == pytest.approx(0.91)

    def test_nan_report_falls_back_to_cells(self):
        cells = [SimpleNamespace(confidence=0.8), SimpleNamespace(confidence=0.6)]
        result = SimpleNamespace(
            confidence=SimpleNamespace(ocr_score=float("nan")),
            pages=[SimpleNamespace(cells=cells)],
        )
        assert q.extract_ocr_confidence(result) == pytest.approx(0.7)

    def test_unavailable_returns_none(self):
        assert q.extract_ocr_confidence(None) is None
        assert q.extract_ocr_confidence(SimpleNamespace(document=object())) is None


# ---------------------------------------------------------------------------
# turkish_ocr_signals — sezgisel sayaçlar
# ---------------------------------------------------------------------------

class TestTurkishSignals:
    def test_i_confusion_detected(self):
        # "kIsa" ve "yIl": küçük harfli kelime içinde büyük I — OCR imzası
        signals = q.turkish_ocr_signals("bu kIsa metin geçen yIl yazıldı")
        assert signals["i_confusion_ratio"] > 0

    def test_clean_text_no_confusion(self):
        signals = q.turkish_ocr_signals("bu kısa metin geçen yıl yazıldı")
        assert signals["i_confusion_ratio"] == 0.0

    def test_vowel_harmony_violation(self):
        # "gidar": i (ön) + a (arka) → ihlal; "araba", "yolda": yalnız arka ünlü
        violating = q.turkish_ocr_signals("gidar")
        assert violating["vowel_harmony_violation_ratio"] == 1.0

        clean = q.turkish_ocr_signals("araba yolda durdu")
        assert clean["vowel_harmony_violation_ratio"] == 0.0

    def test_empty_text(self):
        signals = q.turkish_ocr_signals("")
        assert signals["word_count"] == 0
        assert signals["i_confusion_ratio"] == 0.0


# ---------------------------------------------------------------------------
# MarkdownConverter — cache geriye uyumluluğu (Docling mock'lu)
# ---------------------------------------------------------------------------

def _make_converter() -> MarkdownConverter:
    """Docling DocumentConverter'ı mock'layarak MarkdownConverter kurar."""
    with patch("src.common.parsing.markdown_converter.DocumentConverter"):
        return MarkdownConverter(ocr_engine="easyocr")


def _atoms_cache_path(file_path: Path, ocr_engine: str = "easyocr") -> Path:
    """convert() ile aynı Level-1 cache anahtarını üretir."""
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    ocr_base = f"{file_hash}_{ocr_engine}"
    key = hashlib.md5(ocr_base.encode()).hexdigest()
    return settings.PARSE_CACHE_DIR / f"{key}_atoms.json"


class TestMarkdownConverterQuality:
    def test_old_cache_without_quality_is_still_valid(self, isolated_dirs):
        """quality alanı olmayan eski atoms.json geçerli sayılır, Docling çağrılmaz."""
        pdf = isolated_dirs / "belge.pdf"
        pdf.write_bytes(b"%PDF-fake")

        old_cache = {
            "full_text": "Atom bir\n\nAtom iki",
            "atoms_data": [
                {"text": "Atom bir", "label": "text", "page": 1, "pages": [1]},
                {"text": "Atom iki", "label": "text", "page": 1, "pages": [1]},
            ],
        }
        _atoms_cache_path(pdf).write_text(
            json.dumps(old_cache, ensure_ascii=False), encoding="utf-8"
        )

        conv = _make_converter()
        parsed = conv.convert(str(pdf))

        conv.converter.convert.assert_not_called()  # cache hit — Docling atlanmalı
        assert parsed.quality["atom_count"] == 2
        # Eski cache'te güven skoru yok → None, low_ocr_confidence bayrağı yok
        assert parsed.quality["ocr_mean_confidence"] is None
        assert "low_ocr_confidence" not in parsed.quality["flags"]

    def test_fresh_parse_computes_and_caches_quality(self, isolated_dirs):
        pdf = isolated_dirs / "belge.pdf"
        pdf.write_bytes(b"%PDF-fake-2")

        items = [
            SimpleNamespace(text="Tek atom", label="text",
                            prov=[SimpleNamespace(page_no=1)]),
        ]
        fake_doc = SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items])
        fake_result = SimpleNamespace(
            document=fake_doc,
            confidence=SimpleNamespace(ocr_score=0.42),
            pages=[],
        )

        conv = _make_converter()
        conv.converter.convert.return_value = fake_result
        parsed = conv.convert(str(pdf))

        # 1 atom / 1 sayfa → düşük yoğunluk; güven 0.42 < 0.85 → düşük güven
        assert "low_atom_density" in parsed.quality["flags"]
        assert "low_ocr_confidence" in parsed.quality["flags"]
        assert parsed.quality["ocr_flagged"] is True

        # quality alanı Level-1 atoms.json artefaktına yazılmış olmalı
        cached = json.loads(_atoms_cache_path(pdf).read_text(encoding="utf-8"))
        assert cached["quality"]["ocr_flagged"] is True
        assert cached["quality"]["ocr_mean_confidence"] == pytest.approx(0.42)

        # Sonraki convert: cache hit, güven skoru cache'ten korunur
        conv2 = _make_converter()
        parsed2 = conv2.convert(str(pdf))
        conv2.converter.convert.assert_not_called()
        assert parsed2.quality["ocr_mean_confidence"] == pytest.approx(0.42)
        assert "low_ocr_confidence" in parsed2.quality["flags"]


# ---------------------------------------------------------------------------
# DoclingManager.pack — ocr_flagged chunk metadata'ya taşınır
# ---------------------------------------------------------------------------

def _make_manager() -> DoclingManager:
    with patch("src.common.parsing.markdown_converter.DocumentConverter"):
        return DoclingManager()


def _parsed_doc(ocr_flagged: bool) -> ParsedDocument:
    atoms = [
        {"text": "Atom bir", "label": "text", "page": 1, "pages": [1]},
        {"text": "Atom iki", "label": "text", "page": 1, "pages": [1]},
    ]
    return ParsedDocument(
        full_text="Atom bir\n\nAtom iki",
        atoms=atoms,
        dl_doc=None,
        ocr_base="testhash_easyocr",
        quality={"ocr_flagged": ocr_flagged, "flags": ["low_atom_density"] if ocr_flagged else []},
    )


class TestPackOcrFlag:
    def test_greedy_pack_carries_flag(self, isolated_dirs):
        mgr = _make_manager()
        _, chunks = mgr.pack(_parsed_doc(True), "belge.pdf", min_chars=5, max_chars=50)
        assert chunks
        assert all(c["metadata"]["ocr_flagged"] is True for c in chunks)

    def test_unflagged_document(self, isolated_dirs):
        mgr = _make_manager()
        _, chunks = mgr.pack(_parsed_doc(False), "belge.pdf", min_chars=5, max_chars=50)
        assert all(c["metadata"]["ocr_flagged"] is False for c in chunks)

    def test_parsed_without_quality_defaults_false(self, isolated_dirs):
        """Eski çağıranlar quality alanı vermese de pack kırılmaz."""
        mgr = _make_manager()
        parsed = _parsed_doc(False)
        parsed.quality = {}
        _, chunks = mgr.pack(parsed, "belge.pdf", min_chars=5, max_chars=50)
        assert all(c["metadata"]["ocr_flagged"] is False for c in chunks)

    def test_chunk_cache_hit_injects_flag(self, isolated_dirs):
        """ocr_flagged içermeyen eski Level-2 cache geçerli kalır, bayrak enjekte edilir."""
        parsed = _parsed_doc(True)
        min_chars, max_chars, do_pack = 5, 50, True
        cache_key = hashlib.md5(
            f"{parsed.ocr_base}_{min_chars}_{max_chars}_{do_pack}".encode()
        ).hexdigest()
        old_cache = {
            "full_text": parsed.full_text,
            "chunks": [
                {
                    "text": "Atom bir\n\nAtom iki",
                    "span": [0, 18],
                    "metadata": {"source": "belge.pdf", "page": 1, "pages": [1]},
                }
            ],
        }
        (settings.PARSE_CACHE_DIR / f"{cache_key}.json").write_text(
            json.dumps(old_cache, ensure_ascii=False), encoding="utf-8"
        )

        mgr = _make_manager()
        _, chunks = mgr.pack(
            parsed, "belge.pdf", min_chars=min_chars, max_chars=max_chars, do_pack=do_pack
        )
        assert chunks[0]["metadata"]["ocr_flagged"] is True


# ---------------------------------------------------------------------------
# DocumentManifest — opsiyonel quality_json
# ---------------------------------------------------------------------------

class TestManifestQuality:
    def _doc(self) -> DocumentInput:
        return DocumentInput(
            document_id="doc-1",
            document_type="tutanak",
            collection_name="col-1",
            content_hash="hash-1",
        )

    def test_quality_roundtrip(self, tmp_path):
        manifest = DocumentManifest(db_path=tmp_path / "manifest.db")
        manifest.upsert(self._doc(), status="done", chunk_count=3,
                        quality={"ocr_flagged": True})
        record = manifest.get("doc-1", "col-1")
        assert json.loads(record.quality_json) == {"ocr_flagged": True}

    def test_quality_preserved_when_omitted(self, tmp_path):
        manifest = DocumentManifest(db_path=tmp_path / "manifest.db")
        doc = self._doc()
        manifest.upsert(doc, status="done", chunk_count=3, quality={"ocr_flagged": False})
        manifest.upsert(doc, status="done", chunk_count=3)  # quality verilmedi
        record = manifest.get("doc-1", "col-1")
        assert json.loads(record.quality_json) == {"ocr_flagged": False}

    def test_quality_defaults_none(self, tmp_path):
        manifest = DocumentManifest(db_path=tmp_path / "manifest.db")
        manifest.upsert(self._doc(), status="done", chunk_count=3)
        record = manifest.get("doc-1", "col-1")
        assert record.quality_json is None
