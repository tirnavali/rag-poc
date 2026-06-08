"""Tests for src.ui.components.collection_selector module."""
from unittest.mock import MagicMock, patch, call
import pytest

from src.config.collections import CollectionSpec
from src.ui.components.collection_selector import (
    select_collections_interactive,
    select_collections_streamlit,
)


# Mock data for testing
MOCK_COLLECTIONS = [
    {
        "name": "gazete_arsivi",
        "type": "gazete",
        "embedding_model": "nomic-embed-text-v2-moe",
        "count": 5000,
        "spec": MagicMock(spec=CollectionSpec, name="gazete_spec"),
    },
    {
        "name": "tbmm_minutes",
        "type": "tutanak",
        "embedding_model": "nomic-embed-text-v2-moe",
        "count": 2500,
        "spec": MagicMock(spec=CollectionSpec, name="tbmm_spec"),
    },
    {
        "name": "custom_collection",
        "type": "custom",
        "embedding_model": "nomic-embed-text-v2-moe",
        "count": 1000,
        "spec": MagicMock(spec=CollectionSpec, name="custom_spec"),
    },
]

# Single collection for edge case testing
SINGLE_COLLECTION = [
    {
        "name": "single_col",
        "type": "custom",
        "embedding_model": "nomic-embed-text-v2-moe",
        "count": 100,
        "spec": MagicMock(spec=CollectionSpec, name="single_spec"),
    },
]


class TestSelectCollectionsInteractive:
    """Tests for select_collections_interactive() function."""

    def test_select_collections_interactive_parses_csv_input(self):
        """Should parse comma-separated input and return corresponding specs."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                mock_prompt.return_value = "gazete_arsivi,tbmm_minutes"

                result = select_collections_interactive()

                assert isinstance(result, list)
                assert len(result) == 2
                # Verify the specs are the correct objects
                assert result[0] is MOCK_COLLECTIONS[0]["spec"]
                assert result[1] is MOCK_COLLECTIONS[1]["spec"]

    def test_select_collections_interactive_validates_collection_names(self):
        """Should raise ValueError if invalid collection names are provided."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                mock_prompt.return_value = "gazete_arsivi,invalid_collection"

                with pytest.raises(ValueError) as exc_info:
                    select_collections_interactive()

                assert "invalid_collection" in str(exc_info.value)
                assert "Invalid" in str(exc_info.value)

    def test_select_collections_interactive_uses_defaults(self):
        """Should use first two collections as default when user provides empty input."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                mock_prompt.return_value = ""  # Empty user input

                result = select_collections_interactive()

                assert isinstance(result, list)
                assert len(result) == 2
                # Should default to first two collections
                assert result[0] is MOCK_COLLECTIONS[0]["spec"]
                assert result[1] is MOCK_COLLECTIONS[1]["spec"]

    def test_select_collections_interactive_custom_defaults(self):
        """Should use provided defaults when user provides empty input."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                mock_prompt.return_value = ""  # Empty user input

                result = select_collections_interactive(defaults=["custom_collection"])

                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0] is MOCK_COLLECTIONS[2]["spec"]

    def test_select_collections_interactive_single_collection(self):
        """Should handle single collection selection."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = SINGLE_COLLECTION
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                mock_prompt.return_value = "single_col"

                result = select_collections_interactive()

                assert len(result) == 1
                assert result[0] is SINGLE_COLLECTION[0]["spec"]

    def test_select_collections_interactive_whitespace_handling(self):
        """Should strip whitespace from comma-separated input."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            with patch("rich.prompt.Prompt.ask") as mock_prompt:
                # Spaces around names and multiple spaces between
                mock_prompt.return_value = "  gazete_arsivi  ,  tbmm_minutes  "

                result = select_collections_interactive()

                assert len(result) == 2
                assert result[0] is MOCK_COLLECTIONS[0]["spec"]
                assert result[1] is MOCK_COLLECTIONS[1]["spec"]

    def test_select_collections_interactive_no_collections_error(self):
        """Should raise ValueError when no collections are available."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = []

            with pytest.raises(ValueError) as exc_info:
                select_collections_interactive()

            assert "No collections available" in str(exc_info.value)

    def test_select_collections_interactive_rich_import_error(self):
        """Should raise ImportError with helpful message if rich is not installed."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS
            # Patch the builtins import to simulate missing rich
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "rich.prompt" or name == "rich.table":
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError) as exc_info:
                    select_collections_interactive()

                assert "Rich is required" in str(exc_info.value)
                assert "pip install rich" in str(exc_info.value)


class TestSelectCollectionsStreamlit:
    """Tests for select_collections_streamlit() function."""

    def test_select_collections_streamlit_multiselect_widget(self):
        """Should display multiselect widget and return selected collection specs."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = MOCK_COLLECTIONS
                mock_multiselect.return_value = ["gazete_arsivi", "tbmm_minutes"]

                with patch("streamlit.sidebar") as mock_sidebar:
                    result = select_collections_streamlit()

                    assert isinstance(result, list)
                    assert len(result) == 2
                    assert result[0] is MOCK_COLLECTIONS[0]["spec"]
                    assert result[1] is MOCK_COLLECTIONS[1]["spec"]
                    mock_multiselect.assert_called_once()

    def test_select_collections_streamlit_displays_sidebar_metadata(self):
        """Should display selected collection metadata in sidebar."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = MOCK_COLLECTIONS
                mock_multiselect.return_value = ["gazete_arsivi"]

                with patch("streamlit.sidebar.markdown") as mock_markdown:
                    with patch("streamlit.sidebar.caption") as mock_caption:
                        result = select_collections_streamlit()

                        assert len(result) == 1
                        # Verify sidebar was updated
                        mock_markdown.assert_called_once()
                        mock_caption.assert_called_once()
                        caption_text = mock_caption.call_args[0][0]
                        assert "gazete_arsivi" in caption_text
                        assert "5000 chunks" in caption_text

    def test_select_collections_streamlit_no_sidebar_when_empty(self):
        """Should not display sidebar metadata when no collections selected."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = MOCK_COLLECTIONS
                mock_multiselect.return_value = []  # Empty selection

                with patch("streamlit.sidebar.markdown") as mock_markdown:
                    with patch("streamlit.sidebar.caption") as mock_caption:
                        result = select_collections_streamlit()

                        assert result == []
                        # Sidebar should not be called
                        mock_markdown.assert_not_called()
                        mock_caption.assert_not_called()

    def test_select_collections_streamlit_validates_defaults(self):
        """Should raise ValueError if provided defaults don't exist."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS

            with pytest.raises(ValueError) as exc_info:
                select_collections_streamlit(defaults=["nonexistent_collection"])

            assert "Invalid default collection name(s)" in str(exc_info.value)
            assert "nonexistent_collection" in str(exc_info.value)
            assert "Available collections" in str(exc_info.value)

    def test_select_collections_streamlit_valid_defaults(self):
        """Should accept valid defaults and pass them to multiselect."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = MOCK_COLLECTIONS
                mock_multiselect.return_value = ["custom_collection"]

                with patch("streamlit.sidebar"):
                    result = select_collections_streamlit(defaults=["custom_collection"])

                    # Verify multiselect was called with the correct default
                    call_kwargs = mock_multiselect.call_args[1]
                    assert call_kwargs["default"] == ["custom_collection"]
                    assert len(result) == 1
                    assert result[0] is MOCK_COLLECTIONS[2]["spec"]

    def test_select_collections_streamlit_no_collections_error(self):
        """Should call st.error and st.stop when no collections available."""
        with patch("streamlit.error") as mock_error:
            with patch("streamlit.stop") as mock_stop:
                with patch("src.config.collections.get_available_collections") as mock_get:
                    mock_get.return_value = []

                    select_collections_streamlit()

                    mock_error.assert_called_once()
                    assert "koleksiyon bulunamadı" in mock_error.call_args[0][0].lower()
                    mock_stop.assert_called_once()

    def test_select_collections_streamlit_import_error(self):
        """Should raise ImportError with helpful message if streamlit not installed."""
        with patch.dict("sys.modules", {"streamlit": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'streamlit'")):
                with pytest.raises(ImportError) as exc_info:
                    select_collections_streamlit()

                assert "Streamlit is required" in str(exc_info.value)
                assert "pip install streamlit" in str(exc_info.value)

    def test_select_collections_streamlit_multiselect_format_func(self):
        """Should format multiselect options with count and model info."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = MOCK_COLLECTIONS
                mock_multiselect.return_value = []

                with patch("streamlit.sidebar"):
                    select_collections_streamlit()

                    # Verify the format_func is present and working
                    call_kwargs = mock_multiselect.call_args[1]
                    format_func = call_kwargs["format_func"]
                    formatted = format_func("gazete_arsivi")
                    assert "gazete_arsivi" in formatted
                    assert "5000 chunks" in formatted
                    assert "nomic-embed-text-v2-moe" in formatted

    def test_select_collections_streamlit_single_collection(self):
        """Should handle single collection selection."""
        with patch("streamlit.multiselect") as mock_multiselect:
            with patch("src.config.collections.get_available_collections") as mock_get:
                mock_get.return_value = SINGLE_COLLECTION
                mock_multiselect.return_value = ["single_col"]

                with patch("streamlit.sidebar"):
                    result = select_collections_streamlit()

                    assert len(result) == 1
                    assert result[0] is SINGLE_COLLECTION[0]["spec"]

    def test_select_collections_streamlit_multiple_defaults(self):
        """Should validate multiple defaults and raise if any are invalid."""
        with patch("src.config.collections.get_available_collections") as mock_get:
            mock_get.return_value = MOCK_COLLECTIONS

            with pytest.raises(ValueError) as exc_info:
                select_collections_streamlit(
                    defaults=["gazete_arsivi", "invalid1", "invalid2"]
                )

            error_msg = str(exc_info.value)
            assert "invalid1" in error_msg
            assert "invalid2" in error_msg
