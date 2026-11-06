"""Chat page: talk to the agents; recommendations come with their outfits as pictures.

The conversation id lives in the URL (`?thread=...`), so a reload asks the API for the
conversation (including an open question) instead of starting over. Everything shown
comes from the API's response: the page decides nothing about outfits.
"""

from dataclasses import dataclass

import streamlit as st

from outfit.ui.client import APIError, ChatTurn, Conversation, OutfitAPI
from outfit.ui.widgets import show_flash, show_items
from outfit.ui.worn_form import wore_this

WELCOME = "What do you need an outfit for? Tell me the occasion, and where and when if you know."
SESSION_KEYS = ("thread_id", "history")


@dataclass
class Entry:
    """One chat bubble; a bot reply also carries its turn (outfits, gaps, trace)."""

    role: str  # "user" | "assistant"
    text: str
    turn: ChatTurn | None = None


def history_from(conversation: Conversation) -> list[Entry]:
    """Bubbles for a conversation read back from the API (after a page reload).

    Outfits are not part of the saved messages, so restored replies show text only."""
    entries = [Entry(m["role"], m["content"]) for m in conversation.messages]
    if conversation.question:  # not in `messages` until it is answered
        entries.append(Entry("assistant", conversation.question))
    return entries


def turn_summary(turn: ChatTurn) -> str:
    calls = f"{turn.llm_calls} DeepSeek call{'s' if turn.llm_calls != 1 else ''}"
    return f"{' → '.join(turn.path)} · {calls} · {turn.seconds:.1f}s"


def show_turn(api: OutfitAPI, turn: ChatTurn, key: str = "turn") -> None:
    """The structured part of a reply: outfits, items asked for, gaps, how it ran."""
    for n, outfit in enumerate(turn.outfits):
        with st.container(border=True):
            st.markdown("**Top pick**" if n == 0 else "**Another option**")
            show_items(api, outfit.pieces)
            for reason in outfit.reasons:
                st.caption(reason)
            for warning in outfit.warnings:
                st.caption(f"Note: {warning}")
            wore_this(api, outfit.pieces, f"{key}-{n}", "chat")
    if turn.shown:
        show_items(api, turn.shown)
    for gap in turn.gaps:
        st.info(f"Missing from your wardrobe: {gap}")
    with st.expander("How this turn ran"):
        st.caption(turn_summary(turn))


def start_over() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)
    st.query_params.clear()


def restore(api: OutfitAPI) -> None:
    """Fill the session from `?thread=` once per browser session."""
    if "history" in st.session_state:
        return
    st.session_state.history = []
    thread_id = st.query_params.get("thread")
    if not thread_id:
        return
    try:
        conversation = api.conversation(thread_id)
    except APIError as err:
        st.query_params.clear()
        st.warning(f"Couldn't reopen that conversation: {err.detail}")
        return
    st.session_state.thread_id = thread_id
    st.session_state.history = history_from(conversation)


def send(api: OutfitAPI, text: str) -> None:
    st.session_state.history.append(Entry("user", text))
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"), st.spinner("Thinking about your wardrobe..."):
        try:
            turn = api.chat(text, st.session_state.get("thread_id"))
        except APIError as err:
            st.session_state.history.append(Entry("assistant", f"Sorry, that failed: {err.detail}"))
            return
    st.session_state.thread_id = turn.thread_id
    st.query_params["thread"] = turn.thread_id
    st.session_state.history.append(Entry("assistant", turn.text, turn))


def render(api: OutfitAPI) -> None:
    st.title("Outfit chat")
    show_flash()
    restore(api)
    st.sidebar.button("New conversation", on_click=start_over, width="stretch")

    with st.chat_message("assistant"):
        st.markdown(WELCOME)
    for n, entry in enumerate(st.session_state.history):
        with st.chat_message(entry.role):
            st.markdown(entry.text)
            if entry.turn:
                show_turn(api, entry.turn, key=f"turn{n}")

    if text := st.chat_input("e.g. party in Pune tonight at 7, by cab"):
        send(api, text.strip())
        st.rerun()
