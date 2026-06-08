from src.common.text import normalize_tr, extract_relevant_windows


def test_normalize_lowercase():
    assert normalize_tr("İSTANBUL") == "istanbul"


def test_normalize_turkish_i():
    assert normalize_tr("IŞIK") == "ışık"


def test_normalize_strips_punctuation():
    result = normalize_tr("merhaba, dünya!")
    assert "," not in result
    assert "!" not in result


def test_window_returns_text_around_match():
    text = "a " * 100 + "hedef kelime" + " b" * 100
    result = extract_relevant_windows(text, "hedef kelime")
    assert "hedef kelime" in result


def test_window_fallback_when_no_match():
    text = "x" * 500
    result = extract_relevant_windows(text, "nomatch")
    assert len(result) <= 3000
    assert result == text[:3000]


def test_window_respects_max_total():
    text = "hedef " * 5000
    result = extract_relevant_windows(text, "hedef", max_total=500)
    assert len(result) <= 600
