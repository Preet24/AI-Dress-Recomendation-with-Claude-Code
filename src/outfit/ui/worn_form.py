"""The "I wore this" form under a suggested outfit (chat cards, packing days).

A rating is required; issues, a note and a photo are optional. The photo is matched
locally by the API; the History page shows the match and lets the user correct it.
"""

from datetime import date

import streamlit as st

from outfit.schemas import Occasion
from outfit.ui.client import APIError, Issue, Item, OutfitAPI, Rating, WearSource
from outfit.ui.widgets import label, notify

RATINGS: dict[Rating, str] = {"loved": "Loved it", "fine": "Fine", "not_again": "Not again"}
ISSUES: dict[Issue, str] = {
    "too_warm": "too warm",
    "too_cold": "too cold",
    "too_formal": "too formal",
    "too_casual": "too casual",
    "colours": "colours didn't go",
    "uncomfortable": "uncomfortable",
}
NOT_SAID = "(not said)"


def wore_this(
    api: OutfitAPI,
    pieces: list[Item],
    key: str,
    source: WearSource,
    occasion: Occasion | None = None,
    worn_on: date | None = None,
) -> None:
    """An expander with the form; `key` must be unique on the page."""
    today = date.today()
    names = {i.id: i.name for i in pieces}
    with st.expander("I wore this"), st.form(f"worn-{key}"):
        kept = st.multiselect(
            "What you wore",
            list(names),
            default=list(names),
            format_func=names.get,
            key=f"worn-pieces-{key}",
        )
        rating = st.radio(
            "How was it?",
            list(RATINGS),
            index=None,
            format_func=RATINGS.get,
            horizontal=True,
            key=f"worn-rating-{key}",
        )
        issues = st.multiselect(
            "What was off", list(ISSUES), format_func=ISSUES.get, key=f"worn-issues-{key}"
        )
        left, right = st.columns(2)
        day = left.date_input(
            "Worn on", min(worn_on or today, today), max_value=today, key=f"worn-day-{key}"
        )
        choices = [NOT_SAID, *Occasion]
        said = right.selectbox(
            "Occasion",
            choices,
            index=choices.index(occasion) if occasion else 0,
            format_func=label,
            key=f"worn-occasion-{key}",
        )
        note = st.text_input("Note (for you)", key=f"worn-note-{key}")
        photo = st.file_uploader(
            "Photo of you wearing it (optional, stays on this computer)",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"worn-photo-{key}",
        )
        if not st.form_submit_button("Log it"):
            return
        if rating is None or not kept:
            st.error("Pick a rating and at least one piece.")
            return
        occasion_said = None if said == NOT_SAID else Occasion(said)
        worn = None if day == today else day  # today: the API's own clock decides
        try:
            entry = api.log_worn(kept, rating, issues, note.strip(), worn, occasion_said, source)
        except APIError as err:
            st.error(err.detail)
            return
        message = f"Logged: {RATINGS[rating].lower()}."
        if photo is not None:
            with st.spinner("Checking the photo against your wardrobe..."):
                try:
                    api.upload_worn_photo(entry.id, photo.name, photo.getvalue())
                    message += " Check the photo match on the History page."
                except APIError as err:
                    message += f" The photo wasn't kept: {err.detail}"
    notify(message)
    st.rerun()
