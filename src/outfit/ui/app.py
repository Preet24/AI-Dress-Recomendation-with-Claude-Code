"""Streamlit entry point for the outfit recommender UI.

Every page talks to the API, so start it first, e.g.:
    python scripts/serve.py
    streamlit run src/outfit/ui/app.py      # set OUTFIT_API_URL if the API isn't on :8000
"""

import os

import streamlit as st

from outfit.ui import (
    chat_page,
    history_page,
    packing_page,
    profile_page,
    shopping_page,
    wardrobe_page,
)
from outfit.ui.client import DEFAULT_URL, APIError, OutfitAPI


@st.cache_resource
def default_api(url: str) -> OutfitAPI:
    return OutfitAPI(url)


def get_api() -> OutfitAPI:
    """The session's API client; tests put a fake one in `st.session_state.api`."""
    if "api" not in st.session_state:
        st.session_state.api = default_api(os.environ.get("OUTFIT_API_URL", DEFAULT_URL))
    return st.session_state.api


def sidebar_status(api: OutfitAPI) -> None:
    """Stop with instructions when the API is down: no page works without it."""
    try:
        health = api.health()
    except APIError as err:
        st.error(err.detail)
        st.stop()
    st.sidebar.caption(f"{health.items} items in your wardrobe · {health.model}")


st.set_page_config(page_title="Outfit recommender", layout="wide")
api = get_api()
sidebar_status(api)


def chat() -> None:
    chat_page.render(api)


def wardrobe() -> None:
    wardrobe_page.render(api)


def shopping() -> None:
    shopping_page.render(api)


def packing() -> None:
    packing_page.render(api)


def history() -> None:
    history_page.render(api)


def profile() -> None:
    profile_page.render(api)


st.navigation(
    [
        st.Page(chat, title="Chat", url_path="chat", default=True),
        st.Page(wardrobe, title="Wardrobe", url_path="wardrobe"),
        st.Page(shopping, title="Shopping", url_path="shopping"),
        st.Page(packing, title="Packing", url_path="packing"),
        st.Page(history, title="History", url_path="history"),
        st.Page(profile, title="Profile", url_path="profile"),
    ]
).run()
