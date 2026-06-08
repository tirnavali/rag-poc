"""Collection selection UI components for vector_explorer and chat.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.collections import CollectionSpec


def select_collections_streamlit(defaults: list[str] | None = None) -> list[CollectionSpec]:
    """Streamlit multi-select widget for collection selection.

    Displays a multi-select dropdown in the Streamlit sidebar and shows collection
    metadata (size, embedding model) for each selected collection.

    Args:
        defaults: Default collection names to pre-select. If None, defaults to first 2 collections.

    Returns:
        List of CollectionSpec objects for selected collections.

    Raises:
        ImportError: If streamlit is not installed.
        ValueError: If any provided default collection name does not exist.

    Example:
        >>> specs = select_collections_streamlit(defaults=["gazete_arsivi", "tbmm_minutes"])
    """
    try:
        import streamlit as st
    except ImportError as e:
        raise ImportError(
            "Streamlit is required for select_collections_streamlit(). "
            "Install it with: pip install streamlit"
        ) from e

    from src.config.collections import get_available_collections

    available = get_available_collections()

    if not available:
        st.error("Hiç koleksiyon bulunamadı. Lütfen veri kaynağı ekleyiniz.")
        st.stop()

    # Create set of available names for efficient lookup
    available_names = {col["name"] for col in available}

    # Validate provided defaults
    if defaults:
        invalid_defaults = [d for d in defaults if d not in available_names]
        if invalid_defaults:
            raise ValueError(
                f"Invalid default collection name(s): {', '.join(invalid_defaults)}. "
                f"Available collections: {', '.join(sorted(available_names))}"
            )

    # Create choice labels with collection size
    choice_labels = {
        col["name"]: f"{col['name']} ({col['count']} chunks, {col['embedding_model']})"
        for col in available
    }

    # Set defaults to first two collections if not provided
    default_selected = defaults or [col["name"] for col in available[:2]]

    # Multi-select widget
    selected = st.multiselect(
        "Koleksiyon Seç",
        options=[col["name"] for col in available],
        default=default_selected,
        format_func=lambda name: choice_labels.get(name, name),
    )

    # Show sidebar metadata for selected collections
    if selected:
        st.sidebar.markdown("### Seçili Koleksiyonlar")
        # Build set for O(1) lookup instead of O(n) per item
        selected_set = set(selected)
        for col in available:
            if col["name"] in selected_set:
                st.sidebar.caption(
                    f"**{col['name']}**: {col['count']} chunks | {col['type']} | {col['embedding_model']}"
                )

    # Return specs for selected collections
    return [col["spec"] for col in available if col["name"] in selected]


def select_collections_interactive(defaults: list[str] | None = None) -> list[CollectionSpec]:
    """Interactive Rich prompt for collection selection in terminal.

    Displays a table of available collections and prompts user for comma-separated
    collection names. Supports interactive input in the terminal UI.

    Args:
        defaults: Default collection names to use if user provides empty input.
                 If None, defaults to first 2 collections.

    Returns:
        List of CollectionSpec objects for selected collections.

    Raises:
        ImportError: If rich is not installed.
        ValueError: If any of the user-provided collection names are invalid.

    Example:
        >>> specs = select_collections_interactive()
        >>> specs = select_collections_interactive(defaults=["gazete_arsivi"])
    """
    try:
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError as e:
        raise ImportError(
            "Rich is required for select_collections_interactive(). "
            "Install it with: pip install rich"
        ) from e

    from src.config.collections import get_available_collections
    from src.ui.views import console

    available = get_available_collections()

    if not available:
        console.print("[red]Hiç koleksiyon bulunamadı. Lütfen veri kaynağı ekleyiniz.[/red]")
        raise ValueError("No collections available")

    # Create lookup set for validation
    available_names = {col["name"] for col in available}

    # Display table of available collections
    table = Table(title="Mevcut Koleksiyonlar")
    table.add_column("Koleksiyon Adı", style="cyan")
    table.add_column("Tür", style="magenta")
    table.add_column("Embedding Model", style="green")
    table.add_column("Chunk Sayısı", style="yellow")

    for col in available:
        table.add_row(
            col["name"],
            col["type"],
            col["embedding_model"],
            str(col["count"]),
        )

    console.print(table)

    # Create default string from first two collections
    default_str = ",".join([col["name"] for col in available[:2]])

    # Prompt user for comma-separated input
    prompt_text = (
        "Koleksiyonları seçin (virgülle ayrılmış, boş = varsayılan)\n"
        f"Örnek: gazete_arsivi,tbmm_minutes"
    )
    user_input = Prompt.ask(prompt_text, default=default_str)

    # Parse comma-separated input
    selected_names = [s.strip() for s in user_input.split(",") if s.strip()]

    # If user provided empty input, use defaults or first two collections
    if not selected_names:
        selected_names = defaults or [col["name"] for col in available[:2]]

    # Validate collection names
    invalid = [n for n in selected_names if n not in available_names]
    if invalid:
        raise ValueError(
            f"Invalid collection name(s): {', '.join(invalid)}. "
            f"Available: {', '.join(sorted(available_names))}"
        )

    # Return specs for selected collections
    return [col["spec"] for col in available if col["name"] in selected_names]
