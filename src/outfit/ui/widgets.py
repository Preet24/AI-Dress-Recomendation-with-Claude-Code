"""Pieces the pages share: cutout pictures (fetched through the API and cached), labels,
and the one-line result shown after a change."""

from enum import Enum

import streamlit as st

from outfit.ui.client import APIError, Item, OutfitAPI


def label(value: Enum | str) -> str:
    """'one_piece' -> 'one piece'."""
    return str(value.value if isinstance(value, Enum) else value).replace("_", " ")


def parse_colors(text: str) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def notify(message: str, kind: str = "success") -> None:
    """Shown once at the top of the next run."""
    st.session_state.flash = (kind, message)


def show_flash() -> None:
    if flash := st.session_state.pop("flash", None):
        kind, message = flash
        {"success": st.success, "info": st.info}[kind](message)


@st.cache_data(show_spinner=False, max_entries=300, ttl=600)
def image_bytes(_api: OutfitAPI, url: str) -> bytes:
    """Cutouts by URL; `_api` is not part of the cache key."""
    return _api.image(url)


def picture(api: OutfitAPI, item: Item, caption: str | None = None) -> None:
    try:
        st.image(image_bytes(api, item.image_url), caption=caption or item.name)
    except APIError:
        st.caption(f"{item.name} (no picture)")


def show_items(api: OutfitAPI, items: list[Item], per_row: int = 4) -> None:
    """Items as a grid of captioned cutouts."""
    for start in range(0, len(items), per_row):
        columns = st.columns(per_row)
        for column, item in zip(columns, items[start : start + per_row], strict=False):
            with column:
                picture(api, item)
