"""History page: outfits you logged, with their photo, the photo check and look scores.

The photo check comes from the API (local CLIP); the user confirms it here: "Use your
grey chinos" swaps the piece (or adds it), and the API checks the photo again. "Ask
DeepSeek about this look" is the only thing on the page that sends the photo anywhere.
"""

from uuid import UUID

import streamlit as st

from outfit.ui.client import APIError, GarmentMatch, Item, LookScores, OutfitAPI, WornEntry
from outfit.ui.widgets import label, notify, show_flash, show_items
from outfit.ui.worn_form import ISSUES, RATINGS

VERDICT_MARKS = {"yours": "✅", "unsure": "❓", "extra": "➕", "unknown": "⚠️"}
METRICS = {
    "fit": "Fit",
    "proportion": "Proportion",
    "colour": "Colour on you",
    "coordination": "Coordination",
    "occasion": "Right for the occasion",
    "polish": "Polish",
}


def corrected(entry: WornEntry, garment: GarmentMatch) -> list[UUID]:
    """The entry's pieces with the photo's best match in place of the logged one."""
    pieces = [p.id for p in entry.pieces if p.id != garment.logged]
    return [*pieces, garment.best] if garment.best not in pieces else pieces


def match_section(api: OutfitAPI, entry: WornEntry, wardrobe: dict[UUID, Item]) -> None:
    if entry.match is None:
        return
    st.markdown("**What the photo shows**")
    for n, garment in enumerate(entry.match.garments):
        st.markdown(f"{VERDICT_MARKS[garment.verdict]} {garment.message}")
        best = wardrobe.get(garment.best) if garment.best else None
        if garment.verdict in ("unsure", "extra") and best is not None:
            verb = "Use" if garment.verdict == "unsure" else "Add"
            if st.button(f"{verb} your {best.name}", key=f"fix-{entry.id}-{n}"):
                try:
                    api.correct_worn(entry.id, corrected(entry, garment))
                except APIError as err:
                    st.error(err.detail)
                    return
                notify(f"Updated: you wore your {best.name}")
                st.rerun()
    unseen = [wardrobe[i].name for i in entry.match.unchecked if i in wardrobe]
    if unseen:
        st.caption(f"Not checked (the photo can't show them): {', '.join(unseen)}")


def look_section(api: OutfitAPI, entry: WornEntry) -> None:
    if entry.look is not None:
        show_look(entry.look)
        return
    st.caption("Sends a copy of this photo (no EXIF) to DeepSeek once; it scores the outfit only.")
    if st.button("Ask DeepSeek about this look", key=f"look-{entry.id}"):
        with st.spinner("Asking DeepSeek..."):
            try:
                api.ask_look(entry.id)
            except APIError as err:
                st.error(err.detail)
                return
        notify("DeepSeek scored this look")
        st.rerun()


def show_look(look: LookScores) -> None:
    rows = [
        {"": name, "score": f"{getattr(look, m).score}/5", "why": getattr(look, m).reason}
        for m, name in METRICS.items()
    ]
    st.table(rows)
    for tip in look.tips:
        st.markdown(f"- Tip: {tip}")


def photo_section(api: OutfitAPI, entry: WornEntry, wardrobe: dict[UUID, Item]) -> None:
    if entry.photo_url is None:
        with st.form(f"photo-{entry.id}"):
            photo = st.file_uploader(
                "Add a photo of you wearing it (stays on this computer)",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"photo-file-{entry.id}",
            )
            if st.form_submit_button("Check photo") and photo is not None:
                with st.spinner("Checking the photo against your wardrobe..."):
                    try:
                        api.upload_worn_photo(entry.id, photo.name, photo.getvalue())
                    except APIError as err:
                        st.error(err.detail)
                        return
                st.rerun()
        return
    try:
        st.image(api.image(entry.photo_url), width=320)  # not cached: a new photo, same URL
    except APIError:
        st.caption("(photo missing)")
    match_section(api, entry, wardrobe)
    look_section(api, entry)
    if st.button("Remove photo", key=f"unphoto-{entry.id}"):
        try:
            api.delete_worn_photo(entry.id)
        except APIError as err:
            st.error(err.detail)
            return
        notify("Photo removed")
        st.rerun()


def delete_control(api: OutfitAPI, entry: WornEntry) -> None:
    key = str(entry.id)
    if st.session_state.get("confirm_forget") != key:
        if st.button("Delete", key=f"forget-{key}"):
            st.session_state.confirm_forget = key
            st.rerun()
        return
    st.warning("Delete this entry and its photo? Its feedback stops counting.")
    yes, cancel = st.columns(2)
    if yes.button("Yes, delete", key=f"forget-yes-{key}", type="primary"):
        try:
            api.delete_worn(entry.id)
        except APIError as err:
            st.error(err.detail)
            return
        st.session_state.pop("confirm_forget")
        notify("Deleted the entry")
        st.rerun()
    if cancel.button("Cancel", key=f"forget-cancel-{key}"):
        st.session_state.pop("confirm_forget")
        st.rerun()


def entry_card(api: OutfitAPI, entry: WornEntry, wardrobe: dict[UUID, Item]) -> None:
    with st.container(border=True):
        when = f"{entry.worn_on:%a %d %b %Y}"
        occasion = f" · {label(entry.occasion)}" if entry.occasion else ""
        st.markdown(f"**{when}**{occasion} · {RATINGS[entry.rating]}")
        if entry.issues:
            st.caption("What was off: " + ", ".join(ISSUES[i] for i in entry.issues))
        if entry.note:
            st.caption(entry.note)
        show_items(api, entry.pieces)
        if entry.removed:
            st.caption(f"{entry.removed} piece(s) since removed from your wardrobe")
        photo_section(api, entry, wardrobe)
        delete_control(api, entry)


def render(api: OutfitAPI) -> None:
    st.title("History")
    show_flash()
    try:
        entries = api.worn()
        wardrobe = {i.id: i for i in api.items()} if entries else {}
    except APIError as err:
        st.error(err.detail)
        return
    if not entries:
        st.info('Nothing logged yet: press "I wore this" under an outfit in Chat or Packing.')
        return
    st.caption(
        f"{len(entries)} outfit{'s' if len(entries) != 1 else ''} logged; "
        "ratings nudge future suggestions"
    )
    for entry in entries:
        entry_card(api, entry, wardrobe)
