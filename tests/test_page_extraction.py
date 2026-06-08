import pytest
from src.common.parsing.docling_manager import greedy_pack_atoms, DoclingManager


def test_greedy_pack_atoms_page_aggregation():
    atoms = [
        {"text": "Atom 1", "page": 1, "pages": [1]},
        {"text": "Atom 2", "page": 2, "pages": [2]},
        {"text": "Atom 3", "page": 2, "pages": [2]},
        {"text": "Atom 4", "page": 3, "pages": [3]},
    ]
    
    # min_chars = 10, max_chars = 30
    # "Atom 1" + "\n\n" + "Atom 2" = 6 + 2 + 6 = 14 chars (>= 10)
    # Next is "Atom 3", proposed len = 14 + 2 + 6 = 22 (< 30) -> stays in same chunk
    # Next is "Atom 4", proposed len = 22 + 2 + 6 = 30 -> still fits or splits?
    # Let's test greedy_pack_atoms directly
    packed = greedy_pack_atoms(atoms, min_chars=10, max_chars=25, join_str="\n\n")
    
    assert len(packed) > 0
    for chunk in packed:
        assert "page" in chunk
        assert "pages" in chunk
        assert isinstance(chunk["pages"], list)
        if chunk["pages"]:
            assert chunk["page"] == chunk["pages"][0]


def test_min_token_merge_page_merging():
    class DummyTokenizer:
        def count_tokens(self, text):
            return len(text.split())

    manager = DoclingManager(tokenizer_name=None, max_chunk_tokens=100, min_chunk_tokens=10)
    
    chunks = [
        {
            "text": "Short chunk one",
            "span": (0, 15),
            "metadata": {
                "pages": [5],
                "page": 5,
            }
        },
        {
            "text": "Short chunk two which is also very short",
            "span": (17, 57),
            "metadata": {
                "pages": [5, 6],
                "page": 5,
            }
        },
        {
            "text": "This is a longer chunk that has plenty of tokens to not merge further",
            "span": (59, 127),
            "metadata": {
                "pages": [7],
                "page": 7,
            }
        }
    ]
    
    tokenizer = DummyTokenizer()
    merged = manager._min_token_merge(chunks, tokenizer)
    
    # "Short chunk one" has 3 tokens (< 10 min_chunk_tokens) -> merged with chunk 2
    # Merged chunk has text "Short chunk one\n\nShort chunk two..."
    # and merged pages [5, 6]
    assert len(merged) == 2
    assert merged[0]["metadata"]["pages"] == [5, 6]
    assert merged[0]["metadata"]["page"] == 5
    assert merged[1]["metadata"]["pages"] == [7]
    assert merged[1]["metadata"]["page"] == 7
