from src.common.chunking import build_text_splitter


def test_splitter_creates_chunks():
    splitter = build_text_splitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text("a" * 300)
    assert len(chunks) > 1


def test_splitter_chunk_size():
    splitter = build_text_splitter(chunk_size=50, chunk_overlap=0)
    chunks = splitter.split_text("word " * 100)
    for chunk in chunks:
        assert len(chunk) <= 60
