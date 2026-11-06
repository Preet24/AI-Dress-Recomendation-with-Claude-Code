"""Wardrobe page: browse your garments, add photos, fix tags, remove items.

Every change goes through the API, so it gets the same checks as review_wardrobe.py.
After a change the page reruns and shows a one-line result at the top.
"""

from enum import Enum
from typing import Any

import streamlit as st

from outfit.schemas import Category, Fit, Length, Occasion, Pattern
from outfit.ui.client import APIError, Item, Mode, OutfitAPI
from outfit.ui.widgets import label, notify, parse_colors, picture, show_flash

PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]
MODES: dict[Mode, str] = {
    "flat": "Garment alone (flat-lay, on a hanger)",
    "worn": "Worn by a person (can give a top and a bottom)",
}


def tag_changes(item: Item, values: dict[str, Any]) -> dict[str, Any]:
    """Only the fields the form changed, as the API expects them (enums as values)."""
    before = item.model_dump(mode="json")
    after = item.model_copy(update=values).model_dump(mode="json")
    return {name: after[name] for name in values if after[name] != before[name]}


def summary(item: Item) -> str:
    occasions = ", ".join(label(o) for o in item.occasions) or "no occasions"
    return (
        f"{label(item.category)} · formality {item.formality}/5 · "
        f"warmth {item.warmth}/5 · {occasions}"
    )


def added_message(items: list[Item]) -> str:
    names = ", ".join(i.name for i in items)
    return f"Added {len(items)} garment{'s' if len(items) != 1 else ''}: {names}"


def choice(name: str, options: list, current: Enum, key: str) -> Any:
    return st.selectbox(name, options, index=options.index(current), format_func=label, key=key)


def upload_section(api: OutfitAPI) -> None:
    with st.expander("Add a photo", expanded=not st.session_state.get("has_items", True)):
        # A new key after each upload clears the file picker.
        key = f"photo-{st.session_state.get('uploads', 0)}"
        photo = st.file_uploader("Photo of your clothes", type=PHOTO_TYPES, key=key)
        mode = st.radio("The photo shows", list(MODES), format_func=MODES.get, horizontal=True)
        if st.button("Add to wardrobe", disabled=photo is None, type="primary"):
            with st.spinner("Cutting out, tagging and indexing (about half a minute)..."):
                try:
                    items = api.upload(photo.name, photo.getvalue(), mode)
                except APIError as err:
                    st.error(err.detail)
                    return
            st.session_state.uploads = st.session_state.get("uploads", 0) + 1
            notify(added_message(items))
            st.rerun()


def edit_form(api: OutfitAPI, item: Item) -> None:
    key = str(item.id)
    with st.form(f"edit-{key}"):
        values = {
            "category": choice("Category", list(Category), item.category, f"category-{key}"),
            "subtype": st.text_input("Type", item.subtype, key=f"subtype-{key}").strip(),
            "primary_color_name": st.text_input(
                "Color", item.primary_color_name, key=f"color-{key}"
            ).strip(),
            "secondary_colors": parse_colors(
                st.text_input(
                    "Other colors (comma-separated)",
                    ", ".join(item.secondary_colors),
                    key=f"colors-{key}",
                )
            ),
            "pattern": choice("Pattern", list(Pattern), item.pattern, f"pattern-{key}"),
            "fabric": st.text_input("Fabric", item.fabric, key=f"fabric-{key}").strip(),
            "length": choice("Length", list(Length), item.length, f"length-{key}"),
            "fit": choice("Fit", list(Fit), item.fit, f"fit-{key}"),
            "formality": st.slider(
                "Formality (1 gym, 5 black tie)", 1, 5, item.formality, key=f"formality-{key}"
            ),
            "warmth": st.slider(
                "Warmth (1 very light, 5 heavy winter)", 1, 5, item.warmth, key=f"warmth-{key}"
            ),
            "flowy": st.checkbox("Flowy (catches the wind)", item.flowy, key=f"flowy-{key}"),
            "occasions": st.multiselect(
                "Occasions", list(Occasion), item.occasions, format_func=label, key=f"occ-{key}"
            ),
        }
        if st.form_submit_button("Save changes"):
            changes = tag_changes(item, values)
            if not changes:
                st.info("Nothing changed")
                return
            try:
                saved = api.edit(item.id, changes)
            except APIError as err:
                st.error(err.detail)
                return
            notify(f"Saved {saved.name}: {', '.join(label(c) for c in changes)}")
            st.rerun()


def retag_form(api: OutfitAPI, item: Item) -> None:
    with st.form(f"retag-{item.id}"):
        note = st.text_input(
            "Or describe it, and the tagger redoes every tag",
            placeholder="it's a midi skirt, not shorts",
            key=f"note-{item.id}",
        ).strip()
        if st.form_submit_button("Re-tag") and note:
            with st.spinner("Re-tagging..."):
                try:
                    saved = api.retag(item.id, note)
                except APIError as err:
                    st.error(err.detail)
                    return
            notify(f"Re-tagged as {saved.name}")
            st.rerun()


def delete_control(api: OutfitAPI, item: Item) -> None:
    key = str(item.id)
    if st.session_state.get("confirm_delete") != key:
        if st.button("Delete", key=f"delete-{key}"):
            st.session_state.confirm_delete = key
            st.rerun()
        return
    st.warning(f"Delete {item.name} and its cutout? This can't be undone.")
    yes, cancel = st.columns(2)
    if yes.button("Yes, delete", key=f"yes-{key}", type="primary"):
        try:
            api.delete(item.id)
        except APIError as err:
            st.error(err.detail)
            return
        st.session_state.pop("confirm_delete")
        notify(f"Deleted {item.name}")
        st.rerun()
    if cancel.button("Cancel", key=f"cancel-{key}"):
        st.session_state.pop("confirm_delete")
        st.rerun()


def item_card(api: OutfitAPI, item: Item) -> None:
    with st.container(border=True):
        picture(api, item)
        st.caption(summary(item))
        with st.expander("Edit"):
            edit_form(api, item)
            retag_form(api, item)
            delete_control(api, item)


def render(api: OutfitAPI, per_row: int = 4) -> None:
    st.title("Your wardrobe")
    show_flash()
    upload_section(api)

    left, right = st.columns(2)
    categories = left.multiselect("Categories", list(Category), format_func=label)
    occasions = right.multiselect("Occasions", list(Occasion), format_func=label)
    try:
        items = api.items(categories, occasions)
    except APIError as err:
        st.error(err.detail)
        return
    st.session_state.has_items = bool(items or categories or occasions)
    if not items:
        st.info(
            "Nothing matches these filters."
            if categories or occasions
            else "No items yet: add a photo above."
        )
        return

    st.caption(f"{len(items)} item{'s' if len(items) != 1 else ''}")
    for start in range(0, len(items), per_row):
        for column, item in zip(st.columns(per_row), items[start : start + per_row], strict=False):
            with column:
                item_card(api, item)
