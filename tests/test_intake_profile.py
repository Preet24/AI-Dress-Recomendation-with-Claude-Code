"""Intake with a profile: saved places ("office tomorrow" -> confirm the usual trip) and
the home city as the default city."""

import json
from datetime import datetime, time

from outfit.agents.graph import build_graph
from outfit.agents.intake import (
    MAX_ASKS,
    SAVED_TRIP,
    IntakePatch,
    adopt_place,
    give_up_asking,
    home_city,
    intake_node,
    question_for,
    trip_offer,
    with_trip_question,
)
from outfit.agents.state import Slots
from outfit.config import Settings
from outfit.profile import DressCode, Profile, SavedPlace
from outfit.schemas import Occasion
from outfit.tools.weather import Transport
from tests.fakes import FakeClient, critic_reply, fake_context, reply, router_reply, stylist_reply
from tests.test_intake import run

SETTINGS = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
NOW = datetime(2026, 10, 17, 16, 0)
OFFICE = SavedPlace(
    name="Office",
    occasion=Occasion.OFFICE,
    dress_code=DressCode.BUSINESS_CASUAL,
    start=time(9, 0),
    hours_away=9,
    hours_outdoors=0.25,
    transport=Transport.CAR,
)
GYM = SavedPlace(name="Gym", occasion=Occasion.SPORTS, transport=Transport.TWO_WHEELER)
PROFILE = Profile(home_city="Pune", places=[OFFICE, GYM])
USUAL = "leaving at 09:00, 9 h away, 0.25 h outdoors, by car or cab"


def patch_reply(**fields):
    return reply(json.dumps(fields))


# --- Pure logic -------------------------------------------------------------


def test_a_named_place_brings_its_occasion_and_dress_code():
    slots, notes = adopt_place(Slots(place="Office"), OFFICE, IntakePatch(place="Office"))

    assert slots.occasion is Occasion.OFFICE and slots.formality == (3, 4)
    assert notes == ["Office's dress code is business casual (formality 3-4), from your profile"]


def test_a_formality_already_known_beats_the_dress_code():
    interview = Slots(occasion=Occasion.OFFICE, formality=(4, 5), place="Office")

    slots, notes = adopt_place(interview, OFFICE, IntakePatch(place="Office", occasion="office"))

    assert slots.formality == (4, 5) and notes == []


def test_a_place_without_a_dress_code_keeps_the_occasions_range():
    slots, notes = adopt_place(Slots(place="Gym"), GYM, IntakePatch(place="Gym"))
    assert slots.occasion is Occasion.SPORTS and slots.formality is None and notes == []


def test_the_offer_holds_only_what_is_still_unknown():
    assert trip_offer(Slots(), OFFICE) == {
        "start": "09:00",
        "hours_away": 9,
        "hours_outdoors": 0.25,
        "transport": Transport.CAR,
    }
    assert trip_offer(Slots(start="tomorrow 09:00", hours_away=11), OFFICE) == {
        "hours_outdoors": 0.25,
        "transport": Transport.CAR,
    }


def test_the_confirmation_is_asked_first_and_covers_what_it_would_fill():
    missing = ["city", "hours_away", "hours_outdoors", "transport"]

    assert with_trip_question(missing, Slots(place="Office"), OFFICE) == [SAVED_TRIP, "city"]
    assert with_trip_question(missing, Slots(place="Gym"), GYM) == [
        SAVED_TRIP,
        "city",
        "hours_away",
        "hours_outdoors",
    ]
    answered = Slots(place="Office", use_saved_trip=False)
    assert with_trip_question(missing, answered, OFFICE) == missing


def test_the_question_names_the_usual_trip_and_what_else_is_needed():
    assert question_for([SAVED_TRIP], Slots(place="Office"), OFFICE) == (
        f"Same as usual for Office: {USUAL}? Say yes, or tell me what's different."
    )
    gym = question_for([SAVED_TRIP, "hours_away"], Slots(place="Gym"), GYM)
    assert gym.startswith("Same as usual for Gym: by scooter or motorbike? About how many")


def test_no_city_means_the_home_city():
    slots, notes = home_city(Slots(occasion=Occasion.CASUAL), PROFILE)
    assert slots.city == "Pune" and notes == [
        "the city is Pune, the home city in your profile (no city was given)"
    ]

    assert home_city(Slots(city="Mumbai"), PROFILE)[0].city == "Mumbai"
    assert home_city(Slots(skip_weather=True), PROFILE)[0].city is None
    assert home_city(Slots(), Profile())[0].city is None


def test_giving_up_on_the_confirmation_assumes_the_usual_trip():
    slots = Slots(occasion=Occasion.OFFICE, city="Pune", place="Office")

    slots, notes = give_up_asking(slots, [SAVED_TRIP], OFFICE)

    assert (slots.hours_away, slots.transport, slots.skip_weather) == (9, Transport.CAR, False)
    assert notes == [f"your usual Office trip: {USUAL}"]


# --- The intake node ----------------------------------------------------------


def node(message: str, *replies, slots: Slots | None = None, asks: int = 0, profile=PROFILE):
    client = FakeClient(*replies)
    state = {"messages": [{"role": "user", "content": message}], "asks": asks}
    if slots:
        state["slots"] = slots
    update = intake_node(state, client=client, settings=SETTINGS, now=NOW, profile=lambda: profile)
    return update, client


def test_the_model_sees_the_saved_places():
    _, client = node("office tomorrow", patch_reply(place="Office", start="tomorrow 09:00"))

    payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert payload["saved_places"] == [
        {"name": "Office", "occasion": "office", "usual_start": "09:00"},
        {"name": "Gym", "occasion": "sports", "usual_start": None},
    ]


def test_without_saved_places_the_request_is_unchanged():
    _, client = node("hi", patch_reply(), profile=Profile())
    assert "saved_places" not in json.loads(client.requests[0]["messages"][1]["content"])


def test_office_tomorrow_asks_to_confirm_the_usual_trip():
    update, _ = node("office tomorrow", patch_reply(place="office", start="tomorrow 09:00"))

    slots = update["slots"]
    assert (slots.place, slots.occasion, slots.formality, slots.city) == (
        "Office",
        Occasion.OFFICE,
        (3, 4),
        "Pune",
    )
    assert update["missing"] == [SAVED_TRIP]
    assert slots.hours_away is None  # not filled before the user says yes


def test_yes_fills_the_trip_and_keeps_the_changes():
    asked = Slots(occasion="office", formality=(3, 4), city="Pune", place="Office")
    update, _ = node(
        "yes, but 11 hours",
        patch_reply(use_saved_trip=True, hours_away=11),
        slots=asked,
        asks=1,
    )

    slots = update["slots"]
    assert update["missing"] == []
    assert (slots.start, slots.hours_away, slots.hours_outdoors, slots.transport) == (
        "09:00",
        11,
        0.25,
        Transport.CAR,
    )
    assert (
        "your usual Office trip: leaving at 09:00, 0.25 h outdoors, by car or cab"
        in (update["assumptions"])
    )


def test_no_falls_back_to_the_usual_questions():
    asked = Slots(occasion="office", formality=(3, 4), city="Pune", place="Office")
    update, _ = node("no", patch_reply(use_saved_trip=False), slots=asked, asks=1)

    assert update["missing"] == ["hours_away", "hours_outdoors", "transport"]
    assert update["slots"].transport is None


def test_an_unknown_place_is_dropped():
    update, _ = node("college at 10", patch_reply(place="College", occasion="casual"))

    assert update["slots"].place is None
    assert SAVED_TRIP not in update["missing"]


def test_giving_up_with_a_saved_place_uses_the_usual_trip():
    asked = Slots(occasion="office", formality=(3, 4), city="Pune", place="Office")
    update, _ = node("hmm", patch_reply(), slots=asked, asks=MAX_ASKS)

    assert update["missing"] == []
    assert update["slots"].hours_away == 9 and not update["slots"].skip_weather


# --- The graph ----------------------------------------------------------------


def test_office_tomorrow_conversation():
    client = FakeClient(
        router_reply(),
        patch_reply(place="Office", start="tomorrow 09:00"),
        patch_reply(use_saved_trip=True),
        stylist_reply(),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context(profile=PROFILE))

    first = run(graph, "t", "office tomorrow")
    [question] = first["__interrupt__"]
    second = run(graph, "t", resume="yes")

    assert question.value == (  # the start came with the message, so it isn't offered
        "Same as usual for Office: 9 h away, 0.25 h outdoors, by car or cab? "
        "Say yes, or tell me what's different."
    )
    slots = second["slots"]
    assert (slots.start, slots.hours_away, slots.transport, slots.formality) == (
        "tomorrow 09:00",
        9,
        Transport.CAR,
        (3, 4),
    )
    stylist_facts = client.requests[3]["messages"][-1]["content"]
    assert "home city in your profile" in stylist_facts and "Office's dress code" in stylist_facts
    assert "your usual Office trip" in stylist_facts
    assert len(client.requests) == 5  # router, 2 extractions, stylist, critic


def test_the_api_chat_uses_the_profile_saved_through_put():
    from fastapi.testclient import TestClient

    from outfit.api.app import create_app
    from tests.fakes import fake_services

    client = FakeClient(router_reply(), patch_reply(place="Office", start="tomorrow 09:00"))
    with TestClient(create_app(fake_services(client=client))) as http:
        http.put("/profile", json=PROFILE.model_dump(mode="json"))
        turn = http.post("/chat", json={"message": "office tomorrow"}).json()

    assert turn["kind"] == "question"
    assert turn["text"].startswith("Same as usual for Office: 9 h away")


def test_the_home_city_means_one_question_less():
    client = FakeClient(
        router_reply(),
        patch_reply(occasion="casual", hours_away=3, hours_outdoors=1, transport="walk"),
        stylist_reply(),
        critic_reply(),
    )
    graph = build_graph(client, SETTINGS, fake_context(profile=Profile(home_city="Pune")))

    out = run(graph, "t", "casual walk around for 3 hours, 1 outside")

    assert "__interrupt__" not in out
    assert out["slots"].city == "Pune"
