"""Packing page: describe a trip, confirm where, when and the weather, then pack.

1. The trip form (city, dates, the everyday plan, evenings in, re-wear limits; 0
   dresses = pack none) and the events list.
   **Check trip** asks the API for the preview; nothing is packed yet.
2. The confirmation card: which place (full name, state and country; pick another if
   the first match is wrong), the dates, and every day and event with its weather.
   Changing the form or the events throws the check away.
3. **Pack for this** packs for exactly the place shown: the pieces to take with the
   days each is worn, then one card per day and event.

State: `packing_events` (the events list), `packing_trip` (the trip as checked),
`packing_matches` (the first check's places), `packing_preview`, `packing_plan`.
No DeepSeek call anywhere on this page.
"""

from datetime import date, timedelta

import streamlit as st

from outfit.schemas import Occasion
from outfit.ui.client import (
    Activity,
    APIError,
    OutfitAPI,
    PackingPlan,
    PlacePick,
    Profile,
    Transport,
    Trip,
    TripDay,
    TripEvent,
    TripPreview,
    TripSlot,
    Wears,
)
from outfit.ui.widgets import label, picture, show_flash
from outfit.ui.worn_form import wore_this

MAX_DAYS = 14
TRANSPORTS: list[Transport] = ["walk", "bike", "two_wheeler", "public_transit", "car"]
SOURCES = {
    "forecast": "forecast",
    "last year": "last year, same dates",
    "not checked": "-",
    "indoors": "-",
}
PIECES_PER_ROW = 5


def forget_check() -> None:
    for key in (
        "packing_trip",
        "packing_matches",
        "packing_preview",
        "packing_plan",
        "packing-place",
    ):
        st.session_state.pop(key, None)


def activity_inputs(key: str, default: Activity) -> dict:
    """Occasion, start, hours away / outdoors and transport, in one row."""
    occasion, start, away, outdoors, transport = st.columns(5)
    return {
        "occasion": occasion.selectbox(
            "Occasion",
            list(Occasion),
            list(Occasion).index(default.occasion),
            format_func=label,
            key=f"{key}-occasion",
        ),
        "start": start.time_input("Leaving at", default.start, key=f"{key}-start"),
        "hours_away": away.number_input(
            "Hours away", 0.5, 24.0, float(default.hours_away), 0.5, key=f"{key}-away"
        ),
        "hours_outdoors": outdoors.number_input(
            "Hours outdoors", 0.0, 24.0, float(default.hours_outdoors), 0.5, key=f"{key}-out"
        ),
        "transport": transport.selectbox(
            "Transport",
            TRANSPORTS,
            TRANSPORTS.index(default.transport),
            format_func=label,
            key=f"{key}-transport",
        ),
    }


def events_section(days: int) -> None:
    events: list[TripEvent] = st.session_state.setdefault("packing_events", [])
    st.markdown("**Extra events** (a dinner, a party, a wedding ...)")
    for n, event in enumerate(events):
        text, remove = st.columns([5, 1])
        text.markdown(
            f"- day {event.day}: {label(event.occasion)} at {event.start:%H:%M}, "
            f"{event.hours_away:g} h away ({event.hours_outdoors:g} h outdoors, "
            f"{label(event.transport)})"
        )
        if remove.button("Remove", key=f"remove-event-{n}"):
            events.pop(n)
            forget_check()
            st.rerun()
    added = st.session_state.get("events_added", 0)
    with st.expander("Add an event"), st.form(f"event-{added}"):
        day = st.number_input("Day", 1, MAX_DAYS, 1, key="event-day")
        fields = activity_inputs(
            "event",
            Activity(
                occasion=Occasion.PARTY,
                start="20:00",
                hours_away=4,
                hours_outdoors=0.5,
                transport="car",
            ),  # fmt: skip
        )
        if st.form_submit_button("Add event"):
            if day > days:
                st.error(f"Day {day} is after the trip's {days} days.")
                return
            if fields["hours_outdoors"] > fields["hours_away"]:
                st.error("Hours outdoors can't be more than hours away.")
                return
            events.append(TripEvent(day=day, **fields))
            st.session_state.events_added = added + 1
            forget_check()
            st.rerun()


def trip_form(api: OutfitAPI, profile: Profile) -> None:
    with st.form("trip"):
        city, first, days = st.columns([3, 2, 1])
        city_name = city.text_input(
            "Town or city",
            placeholder=profile.home_city or "e.g. Panjim, Calangute, Manali",
            help="A town, not a state: 'Goa' finds Genoa. Empty: your home city.",
            key="trip-city",
        )
        first_day = first.date_input(
            "First day", date.today() + timedelta(days=1), key="trip-first"
        )
        n_days = days.number_input("Days", 1, MAX_DAYS, 3, key="trip-days")
        st.markdown("**Every day** (the first and last day as travel, unless unticked)")
        everyday = activity_inputs("everyday", Activity())
        travel = st.checkbox("First and last day are travel days", True, key="trip-travel")
        evenings = st.checkbox(
            "Evenings in (relax / sleep)",
            False,
            key="trip-evenings",
            help="Something to lounge and sleep in every night, at home: no shoes, no weather",
        )
        st.markdown("**Wear one piece on up to ... days**")
        tops, bottoms, dresses = st.columns(3)
        wears = Wears(
            tops=tops.number_input("Tops", 1, MAX_DAYS, 2, key="wears-tops"),
            bottoms=bottoms.number_input("Bottoms", 1, MAX_DAYS, 3, key="wears-bottoms"),
            one_pieces=dresses.number_input(
                "Dresses / one-pieces",
                0,
                MAX_DAYS,
                1,
                key="wears-one",
                help="0: don't pack any dresses or one-pieces",
            ),  # fmt: skip
        )
        if not st.form_submit_button("Check trip", type="primary"):
            return
    if everyday["hours_outdoors"] > everyday["hours_away"]:
        st.error("Hours outdoors can't be more than hours away.")
        return
    trip = Trip(
        city=city_name.strip() or None,
        first_day=first_day,
        days=n_days,
        everyday=Activity(**everyday),
        events=st.session_state.get("packing_events", []),
        travel_days=travel,
        evenings_in=evenings,
        wears=wears,
    )
    forget_check()
    try:
        preview = api.preview_trip(trip)
    except APIError as err:
        st.error(err.detail)
        return
    st.session_state.packing_trip = trip
    st.session_state.packing_matches = preview.matches
    st.session_state.packing_preview = preview
    st.rerun()


def choose_place(api: OutfitAPI, trip: Trip, preview: TripPreview) -> None:
    matches: list[PlacePick] = st.session_state.packing_matches
    if len(matches) < 2:
        return
    labels = [m.label for m in matches]
    shown = labels.index(preview.place.label) if preview.place else 0
    picked = st.radio("Which place?", labels, shown, key="packing-place")
    if picked != labels[shown]:
        place = matches[labels.index(picked)]
        try:
            st.session_state.packing_preview = api.preview_trip(
                trip.model_copy(update={"place": place})
            )
        except APIError as err:
            st.error(err.detail)
            return
        st.session_state.pop("packing_plan", None)
        st.rerun()


def confirmation(api: OutfitAPI, trip: Trip, preview: TripPreview) -> None:
    with st.container(border=True):
        st.subheader("Is this right?")
        choose_place(api, trip, preview)
        where = preview.place.label if preview.place else "weather not checked"
        st.markdown(f"**Where:** {where}  \n**When:** {preview.dates}")
        st.table(
            [
                {
                    "Day": row.slot.day,
                    "When": f"{row.slot.start:%a %d %b %H:%M}",
                    "Occasion": slot_title(row.slot),
                    "Out": "in"
                    if row.slot.indoors
                    else f"{row.slot.hours_away:g} h ({row.slot.hours_outdoors:g} outdoors)",
                    "By": "-" if row.slot.indoors else label(row.slot.transport),
                    "Weather": row.weather,
                    "From": SOURCES[row.source],
                }
                for row in preview.rows
            ]
        )
        w = preview.wears
        one_piece = f"a dress or one-piece {w.one_pieces}" if w.one_pieces else "no dresses"
        st.caption(f"Wear a top on up to {w.tops} days, a bottom {w.bottoms}, {one_piece}.")
        for assumption in preview.assumptions:
            st.caption(f"Note: {assumption}")
        if trip.city and preview.place is None:
            st.warning("No place found by that name: try a town or city nearby.")
            return
        if st.button("Pack for this", type="primary", key="pack"):
            confirmed = trip.model_copy(update={"place": preview.place})
            with st.spinner("Packing..."):
                try:
                    st.session_state.packing_plan = api.pack_trip(confirmed)
                except APIError as err:
                    st.error(err.detail)
                    return
            st.rerun()


def slot_title(slot: TripSlot) -> str:
    return "Evening in (lounge / sleep)" if slot.indoors else label(slot.occasion)


def day_card(api: OutfitAPI, day: TripDay) -> None:
    slot = day.slot
    with st.container(border=True):
        st.markdown(f"**Day {slot.day} · {slot_title(slot)}** · {slot.start:%a %d %b %H:%M}")
        st.caption(day.weather)
        if day.outfit is None:
            st.warning("Nothing in your wardrobe fits this one (see what's missing).")
            return
        for column, piece in zip(st.columns(PIECES_PER_ROW), day.outfit.pieces, strict=False):
            with column:
                picture(api, piece)
        for note in [*day.notes, *day.outfit.warnings]:
            st.warning(note)
        with st.expander("Why this outfit"):
            for reason in day.outfit.reasons:
                st.markdown(f"- {reason}")
        wore_this(
            api,
            day.outfit.pieces,
            key=slot.name.replace(" ", "-"),
            source="packing",
            occasion=slot.occasion,
            worn_on=slot.start.date(),
        )


def show_plan(api: OutfitAPI, plan: PackingPlan) -> None:
    where = f" for {plan.place}" if plan.place else ""
    st.subheader(f"Pack {len(plan.pieces)} pieces{where}")
    st.caption(plan.dates)
    for start in range(0, len(plan.pieces), PIECES_PER_ROW):
        row = plan.pieces[start : start + PIECES_PER_ROW]
        for column, packed in zip(st.columns(PIECES_PER_ROW), row, strict=False):
            with column:
                picture(api, packed.item)
                st.caption(f"day{'s' * (len(packed.days) > 1)} {', '.join(map(str, packed.days))}")
    for note in plan.notes:
        st.markdown(f"- {note}")
    if plan.gaps:
        st.markdown("**Missing from your wardrobe**")
        for gap in plan.gaps:
            st.markdown(f"- {gap}")
    for assumption in plan.assumptions:
        st.caption(f"Note: {assumption}")
    for day in plan.days:
        day_card(api, day)


def render(api: OutfitAPI) -> None:
    st.title("Packing")
    show_flash()
    try:
        profile = api.profile()
    except APIError as err:
        st.error(err.detail)
        return
    trip_form(api, profile)
    events_section(st.session_state.get("trip-days", 3))
    trip, preview = st.session_state.get("packing_trip"), st.session_state.get("packing_preview")
    if trip and preview:
        confirmation(api, trip, preview)
    if plan := st.session_state.get("packing_plan"):
        show_plan(api, plan)
