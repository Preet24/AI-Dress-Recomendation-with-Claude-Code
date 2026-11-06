"""Shopping page: what should I buy, and should I buy this?

The occasions you tick (plus your saved places) are what both tabs plan for.

- What should I buy? lists what the wardrobe is missing for the occasions chosen there
  and common pieces that would help most (local rules, no DeepSeek call); the last
  result stays in `st.session_state.shopping_suggest`.
- Should I buy this? cuts out and tags a photo (one DeepSeek call per garment) but
  stores nothing; each garment found gets a report card, and "Add to wardrobe" stores
  it if you bought it. The last check stays in `st.session_state.shopping_check`.
"""

import streamlit as st

from outfit.schemas import Occasion
from outfit.ui.client import (
    APIError,
    Mode,
    OutfitAPI,
    Pairing,
    Profile,
    ShoppingCheck,
    ShoppingReport,
    ShoppingSuggestions,
    Suggestion,
)
from outfit.ui.widgets import label, notify, picture, show_flash

PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]
MODES: dict[Mode, str] = {
    "worn": "Worn by a model or a person",
    "flat": "The garment alone (product shot, on a hanger)",
}
VERDICTS = {
    "buy": (st.success, "Buy: it earns its place"),
    "maybe": (st.info, "Maybe: it works, but changes little"),
    "skip": (st.warning, "Skip"),
}
PAIRS_PER_ROW = 5


def occasions_form(api: OutfitAPI, profile: Profile) -> None:
    with st.form("occasions"):
        places = ", ".join(f"{p.name} ({label(p.occasion)})" for p in profile.places)
        chosen = st.multiselect(
            "Occasions you dress for",
            list(Occasion),
            profile.occasions,
            format_func=label,
            key="occasions",
            help="With none ticked and no saved places: casual, office and party.",
        )
        if places:
            st.caption(f"Your saved places always count too: {places}.")
        if st.form_submit_button("Save occasions"):
            try:
                api.save_profile(profile.model_copy(update={"occasions": chosen}))
            except APIError as err:
                st.error(err.detail)
                return
            notify("Saved the occasions you dress for")
            st.rerun()


def check_section(api: OutfitAPI) -> None:
    # A new key after each check clears the file picker.
    key = f"shop-photo-{st.session_state.get('shop_checks', 0)}"
    photo = st.file_uploader("Photo of the item (shop photo or screenshot)", PHOTO_TYPES, key=key)
    mode = st.radio("The photo shows", list(MODES), format_func=MODES.get, horizontal=True)
    if st.button("Check it", disabled=photo is None, type="primary", key="check"):
        with st.spinner("Cutting out, tagging and comparing with your wardrobe..."):
            try:
                result = api.check_item(photo.name, photo.getvalue(), mode)
            except APIError as err:
                st.error(err.detail)
                return
        st.session_state.shopping_check = result
        st.session_state.shop_checks = st.session_state.get("shop_checks", 0) + 1
        st.rerun()


def pair_grid(api: OutfitAPI, pairs: list[Pairing]) -> None:
    for column, pair in zip(st.columns(PAIRS_PER_ROW), pairs, strict=False):
        with column:
            picture(api, pair.item)
            st.caption(pair.reason)


def report_card(api: OutfitAPI, report: ShoppingReport) -> None:
    item = report.candidate
    with st.container(border=True):
        left, right = st.columns([1, 3])
        with left:
            picture(api, item)
        with right:
            show, text = VERDICTS[report.verdict]
            show(f"**{text}** · {item.name}")
            for reason in report.reasons:
                st.markdown(f"- {reason}")
            if report.unlocks:
                st.caption("Makes outfits possible for: " + ", ".join(report.unlocks))
            added = st.session_state.get("shop_added", set())
            if report.owned:
                st.caption("Already in your wardrobe.")
            elif str(item.id) in added:
                st.caption("Added to your wardrobe.")
            elif st.button("Add to wardrobe (I bought it)", key=f"add-{item.id}"):
                try:
                    stored = api.add_candidate(item.id)
                except APIError as err:
                    st.error(err.detail)
                    return
                st.session_state.shop_added = added | {str(item.id)}
                notify(f"Added {stored.name} to your wardrobe")
                st.rerun()
        if report.goes_with:
            st.markdown("**Goes with**")
            pair_grid(api, report.goes_with)
        if report.clashes:
            st.markdown("**Clashes with**")
            pair_grid(api, report.clashes)
        if report.looks_like and not report.owned:
            st.markdown(f"**Closest piece you own** (similarity {report.looks_like.score:.2f})")
            pair_grid(api, [report.looks_like])


def show_check(api: OutfitAPI, result: ShoppingCheck) -> None:
    if not result.reports:
        st.info("No garment found in that photo.")
        return
    st.caption(f"Checked {len(result.checked)} situations; weather: {result.climate}.")
    with st.expander("What was checked"):
        st.write(", ".join(result.checked))
    for report in result.reports:
        report_card(api, report)


def suggest_section(api: OutfitAPI, profile: Profile) -> None:
    usual = list(dict.fromkeys([*profile.occasions, *(p.occasion for p in profile.places)]))
    chosen = st.multiselect(
        "Shop for",
        list(Occasion),
        usual,
        format_func=label,
        key="shop_for",
        help="Empty: the occasions you dress for and your saved places.",
    )
    if st.button("Suggest", type="primary", key="suggest"):
        with st.spinner("Trying common pieces against your wardrobe..."):
            try:
                st.session_state.shopping_suggest = api.suggest_purchases(chosen)
            except APIError as err:
                st.error(err.detail)


def show_missing(result: ShoppingSuggestions) -> None:
    st.markdown("**What your wardrobe is missing**")
    if not result.no_outfit and not result.gaps:
        st.markdown("- nothing: every occasion and weather has a good outfit")
    for situation in result.no_outfit:
        st.markdown(f"- no outfit for {situation}")
    for gap in result.gaps:
        st.markdown(f"- {gap.gap} ({', '.join(gap.where)})")


def suggestion_card(api: OutfitAPI, suggestion: Suggestion) -> None:
    with st.container(border=True):
        show, text = VERDICTS[suggestion.verdict]
        show(f"**{suggestion.name}** · {text}")
        if suggestion.together_with:
            st.caption(f"Buy it together with the {suggestion.together_with}.")
        elif suggestion.after:
            st.caption(f"Counted as if you already had the {', '.join(suggestion.after)}.")
        for reason in suggestion.reasons:
            st.markdown(f"- {reason}")
        if suggestion.goes_with:
            st.markdown("**Goes with**")
            pair_grid(api, suggestion.goes_with)


def show_suggestions(api: OutfitAPI, result: ShoppingSuggestions) -> None:
    st.caption(f"Checked {len(result.checked)} situations; weather: {result.climate}.")
    show_missing(result)
    if not result.suggestions:
        st.info("Nothing to buy: no common piece would add much to your wardrobe.")
    for suggestion in result.suggestions:
        suggestion_card(api, suggestion)


def render(api: OutfitAPI) -> None:
    st.title("Shopping")
    show_flash()
    try:
        profile = api.profile()
    except APIError as err:
        st.error(err.detail)
        return
    occasions_form(api, profile)
    what, check = st.tabs(["What should I buy?", "Should I buy this?"])
    with what:
        suggest_section(api, profile)
        if result := st.session_state.get("shopping_suggest"):
            show_suggestions(api, result)
    with check:
        check_section(api)
        if result := st.session_state.get("shopping_check"):
            show_check(api, result)
