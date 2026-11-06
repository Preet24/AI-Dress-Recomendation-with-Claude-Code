import json
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from outfit.agents.graph import build_graph
from outfit.evals.fixtures import WARDROBE_PATH, load_wardrobe, restore, snapshot, weather_fetch
from outfit.tools.weather import Outing, Transport, weather_for_outing
from tests.fakes import FakeClient, critic_reply, reply, router_reply, stylist_reply
from tests.test_agent_flow import SETTINGS, WARDROBE
from tests.test_router_refine import NO_SHOES

NOW = datetime(2026, 10, 17, 16, 0)


def advice_for(weather, transport, hours_outdoors, start=NOW, hours_away=3):
    outing = Outing(
        start=start, hours_away=hours_away, hours_outdoors=hours_outdoors, transport=transport
    )
    return weather_for_outing("Pune", outing, weather_fetch(weather))


def test_warm_dry_needs_nothing_special():
    place, summary, advice = advice_for("warm_dry", Transport.CAR, 0.5)

    assert place.name == "Pune" and place.timezone == "Asia/Kolkata"
    assert advice.warmth == (1, 2) and not advice.needs_outerwear and advice.notes == []


def test_rain_on_a_scooter_needs_a_rain_layer_and_no_flowy():
    _, summary, advice = advice_for("rainy", Transport.TWO_WHEELER, 1)

    assert summary.rain_chance_max == 80
    assert advice.effective_temp == 19 and advice.warmth == (1, 3)
    assert advice.needs_outerwear and advice.avoid_flowy


def test_cold_outside_needs_warm_outerwear():
    _, _, advice = advice_for("cold", Transport.WALK, 2)

    assert advice.warmth == (2, 5) and advice.outerwear_min_warmth == 4


def test_fixture_covers_every_hour_even_past_midnight():
    _, summary, _ = advice_for("warm_dry", Transport.CAR, 0.5, NOW.replace(hour=22), 4)

    assert (summary.start, summary.end) == (NOW.replace(hour=22), datetime(2026, 10, 18, 2, 0))


def test_snapshot_round_trips_items_and_vectors():
    from tests.fakes import fake_context

    store = fake_context(WARDROBE).store

    copy = restore(json.loads(json.dumps(snapshot(store))))

    assert sorted(copy.list_items(), key=str) == sorted(store.list_items(), key=str)
    for item in store.list_items():
        assert np.allclose(copy.vector(item.id), store.vector(item.id))


@pytest.mark.skipif(not WARDROBE_PATH.exists(), reason="no wardrobe snapshot")
def test_the_wardrobe_snapshot_loads():
    store = load_wardrobe()

    items = store.list_items()
    assert items
    assert all(np.isclose(np.linalg.norm(store.vector(i.id)), 1, atol=1e-3) for i in items)


def test_a_fixed_clock_reaches_intake_and_the_weather():
    from tests.fakes import fake_context

    trip = {"occasion": "casual", "city": "Pune", "start": "now", "hours_away": 3}
    trip |= {"hours_outdoors": 1, "transport": "walk"}
    client = FakeClient(
        router_reply(),
        reply(json.dumps(trip)),
        stylist_reply("Your white t-shirt and medium blue jeans." + NO_SHOES),
        critic_reply(),
    )
    context = replace(fake_context(WARDROBE), fetch=weather_fetch("warm_dry"), now=NOW)
    graph = build_graph(client, SETTINGS, context)

    out = graph.invoke(
        {"messages": [{"role": "user", "content": "casual walk now"}]},
        {"configurable": {"thread_id": "t"}},
    )

    intake_payload = json.loads(client.requests[1]["messages"][1]["content"])
    assert intake_payload["now"] == "Saturday 2026-10-17 16:00"
    assert out["weather"].start == NOW
