"""Profile page: about you, and the places you go with their dress code and usual trip.

Every field is optional. Each save sends the whole profile to the API, which validates
it (e.g. unique place names); its message is shown if it refuses. The photo is only
sent to DeepSeek when the user presses "Analyse my colouring".
"""

from enum import Enum
from typing import Any, get_args

import streamlit as st

from outfit.schemas import Fit, Occasion
from outfit.ui.client import (
    APIError,
    Build,
    Colouring,
    DressCode,
    Gender,
    OutfitAPI,
    Profile,
    SavedPlace,
    Transport,
)
from outfit.ui.widgets import label, notify, parse_colors, show_flash

PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]
PHOTO_NOTE = (
    "Optional. The photo stays on this computer. **Analyse my colouring** sends a smaller "
    "copy (without location or camera data) to DeepSeek once, to judge skin undertone, "
    "hair and eye colour. It never guesses age, ethnicity or body shape; your build comes "
    "from what you enter above."
)
DRESS_CODES: dict[DressCode, str] = {  # outfit.profile.DRESS_CODE_FORMALITY, in words
    "formal": "formal (formality 4-5)",
    "business": "business (4)",
    "business_casual": "business casual (3-4)",
    "smart_casual": "smart casual (3)",
    "casual": "casual (2-3)",
}


def show(value: Enum | str | None) -> str:
    if value is None:
        return "not set"
    return DRESS_CODES.get(str(value), label(value))


def optional_choice(
    where: Any, name: str, options: list, current: Any, key: str, **kwargs: Any
) -> Any:
    """A selectbox (in the column `where`) whose first option, "not set", is None."""
    choices = [None, *options]
    index = choices.index(current) if current in choices else 0
    return where.selectbox(name, choices, index=index, format_func=show, key=key, **kwargs)


def text_or_none(text: str) -> str | None:
    return text.strip() or None


def place_summary(place: SavedPlace) -> str:
    """'Office · office · business casual (3-4) · 09:00, 9 h, car'."""
    trip = [
        place.start.strftime("%H:%M") if place.start else None,
        f"{place.hours_away:g} h" if place.hours_away else None,
        f"{place.hours_outdoors:g} h outside" if place.hours_outdoors is not None else None,
        label(place.transport) if place.transport else None,
    ]
    parts = [place.name, label(place.occasion)]
    parts += [DRESS_CODES[place.dress_code]] if place.dress_code else []
    parts += [", ".join(t for t in trip if t)] if any(trip) else []
    return " · ".join(parts)


def save(api: OutfitAPI, profile: Profile, message: str) -> None:
    try:
        api.save_profile(profile)
    except APIError as err:
        st.error(err.detail)
        return
    notify(message)
    st.rerun()


def about_form(api: OutfitAPI, profile: Profile) -> None:
    with st.form("about"):
        left, right = st.columns(2)
        values = {
            "gender": optional_choice(
                left, "Gender", list(get_args(Gender)), profile.gender, "gender"
            ),
            "home_city": text_or_none(
                right.text_input("Home city", profile.home_city or "", key="home_city")
            ),
            "height_cm": left.number_input(
                "Height (cm)",
                min_value=100,
                max_value=230,
                value=profile.height_cm,
                step=1,
                placeholder="not set",
                key="height_cm",
            ),
            "build": optional_choice(right, "Build", list(get_args(Build)), profile.build, "build"),
            "waist": text_or_none(
                left.text_input("Waist", profile.waist or "", placeholder="32", key="waist")
            ),
            "top_size": text_or_none(
                right.text_input("Top size", profile.top_size or "", placeholder="M", key="top")
            ),
            "fit_tops": optional_choice(
                left, "Fit for tops", list(Fit), profile.fit_tops, "fit_tops"
            ),
            "fit_bottoms": optional_choice(
                right, "Fit for bottoms", list(Fit), profile.fit_bottoms, "fit_bottoms"
            ),
            "colors_liked": parse_colors(
                st.text_input(
                    "Colors you like (comma-separated)",
                    ", ".join(profile.colors_liked),
                    key="colors_liked",
                )
            ),
            "colors_avoided": parse_colors(
                st.text_input(
                    "Colors you avoid (still used when nothing else fits)",
                    ", ".join(profile.colors_avoided),
                    key="colors_avoided",
                )
            ),
        }
        if st.form_submit_button("Save", type="primary"):
            save(api, profile.model_copy(update=values), "Saved your details")


def place_form(api: OutfitAPI, profile: Profile, place: SavedPlace | None, key: str) -> None:
    """Edit `place`, or add a new one when it is None."""
    old = place or SavedPlace(name="", occasion=Occasion.CASUAL)
    with st.form(key):
        name = st.text_input(
            "Name", old.name, placeholder="Office, Gym, College", key=f"{key}-name"
        )
        left, right = st.columns(2)
        new = SavedPlace(
            name=name.strip(),
            occasion=left.selectbox(
                "Occasion",
                list(Occasion),
                index=list(Occasion).index(old.occasion),
                format_func=label,
                key=f"{key}-occasion",
            ),
            dress_code=optional_choice(
                right,
                "Dress code",
                list(DRESS_CODES),
                old.dress_code,
                f"{key}-dress",
                help="Not set = the occasion's usual formality",
            ),
            start=left.time_input("Usual start", old.start, step=900, key=f"{key}-start"),
            transport=optional_choice(
                right,
                "Usual transport",
                list(get_args(Transport)),
                old.transport,
                f"{key}-transport",
            ),
            hours_away=left.number_input(
                "Hours away",
                min_value=0.25,
                max_value=24.0,
                value=old.hours_away,
                step=0.25,
                placeholder="not set",
                key=f"{key}-away",
            ),
            hours_outdoors=right.number_input(
                "Hours outdoors",
                min_value=0.0,
                max_value=24.0,
                value=old.hours_outdoors,
                step=0.25,
                placeholder="not set",
                key=f"{key}-outdoors",
            ),
        )
        if st.form_submit_button("Save place" if place else "Add place"):
            if not new.name:
                st.error("Give the place a name")
                return
            places = [new if p is place else p for p in profile.places]
            if place is None:
                places.append(new)
                st.session_state.places_added = st.session_state.get("places_added", 0) + 1
            verb = "Saved" if place else "Added"
            save(api, profile.model_copy(update={"places": places}), f"{verb} {new.name}")


def places_section(api: OutfitAPI, profile: Profile) -> None:
    st.subheader("Your places")
    st.caption("Places you go often, with their dress code and usual trip.")
    for place in profile.places:
        key = f"place-{place.name.lower()}"
        with st.expander(place_summary(place)):
            place_form(api, profile, place, key)
            if st.button(f"Remove {place.name}", key=f"{key}-remove"):
                places = [p for p in profile.places if p is not place]
                save(api, profile.model_copy(update={"places": places}), f"Removed {place.name}")
    # A new key after each added place gives an empty form.
    with st.expander("Add a place", expanded=not profile.places):
        place_form(api, profile, None, f"new-place-{st.session_state.get('places_added', 0)}")


def photo_controls(api: OutfitAPI) -> None:
    try:
        photo = api.profile_photo()
    except APIError as err:
        st.error(err.detail)
        return
    left, right = st.columns([1, 3])
    if photo:
        left.image(photo, width=160)
    with right:
        # A new key after each upload clears the file picker.
        key = f"profile-photo-{st.session_state.get('profile_photos', 0)}"
        new = st.file_uploader("A photo of your face in daylight", type=PHOTO_TYPES, key=key)
        if new is not None and st.button("Save photo", key="save-photo"):
            try:
                api.upload_profile_photo(new.name, new.getvalue())
            except APIError as err:
                st.error(err.detail)
                return
            st.session_state.profile_photos = st.session_state.get("profile_photos", 0) + 1
            notify("Saved your photo (only on this computer)")
            st.rerun()
        if not photo:
            return
        if st.button("Analyse my colouring (sends the photo to DeepSeek)", key="analyse"):
            with st.spinner("Looking at your colouring..."):
                try:
                    colouring = api.analyse_colouring()
                except APIError as err:
                    st.error(err.detail)
                    return
            notify(f"Your colouring: {colouring.summary}. Edit it below if it is off.")
            st.rerun()
        if st.button("Delete photo (keeps the colouring text)", key="delete-photo"):
            try:
                api.delete_profile_photo()
            except APIError as err:
                st.error(err.detail)
                return
            notify("Deleted your photo")
            st.rerun()


def colouring_form(api: OutfitAPI, profile: Profile) -> None:
    current = profile.colouring
    with st.form("colouring"):
        summary = st.text_input(
            "Your colouring (leave empty to clear all three)",
            current.summary if current else "",
            placeholder="warm undertone, medium depth, dark brown hair and eyes",
            key="colouring_summary",
        ).strip()
        suits = st.text_input(
            "Colors that suit you (comma-separated)",
            ", ".join(current.suits) if current else "",
            key="colouring_suits",
        )
        go_easy = st.text_input(
            "Colors to go easy on",
            ", ".join(current.go_easy) if current else "",
            key="colouring_go_easy",
        )
        if st.form_submit_button("Save colouring"):
            colouring = (
                Colouring(summary=summary, suits=parse_colors(suits), go_easy=parse_colors(go_easy))
                if summary
                else None
            )
            message = "Saved your colouring" if colouring else "Cleared your colouring"
            save(api, profile.model_copy(update={"colouring": colouring}), message)


def photo_section(api: OutfitAPI, profile: Profile) -> None:
    st.subheader("Photo and colouring")
    st.caption(PHOTO_NOTE)
    photo_controls(api)
    colouring_form(api, profile)


def render(api: OutfitAPI) -> None:
    st.title("Your profile")
    show_flash()
    st.caption("Everything is optional and stays on this computer (data/profile.json).")
    try:
        profile = api.profile()
    except APIError as err:
        st.error(err.detail)
        return
    st.subheader("About you")
    about_form(api, profile)
    places_section(api, profile)
    photo_section(api, profile)
