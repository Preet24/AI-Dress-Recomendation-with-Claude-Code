"""Taste from the profile in outfit ranking: liked / avoided main colors, preferred fits."""

from dataclasses import replace

import pytest

from outfit.agents.orchestrate import orchestrate_node
from outfit.agents.state import Slots
from outfit.evals.fixtures import load_wardrobe
from outfit.profile import Profile
from outfit.schemas import Fit, Occasion
from outfit.tools.outfits import (
    AVOIDED_COLOR,
    LIKED_COLOR,
    PREFERRED_FIT,
    OutfitRequest,
    color_is,
    suggest_outfits,
    taste,
)
from tests.fakes import fake_context
from tests.test_outfits import bottom, shoes, top

CASUAL = OutfitRequest(occasion=Occasion.CASUAL)


def test_a_color_matches_whole_words_of_the_main_color():
    navy = top(name="navy blue")
    assert color_is(navy, frozenset({"navy"})) == "navy"
    assert color_is(navy, frozenset({"navy blue"})) == "navy blue"
    assert color_is(navy, frozenset({"nav", "red"})) is None
    assert color_is(top(name="off-white"), frozenset({"white"})) == "white"
    # secondary colors don't count
    assert color_is(top(name="white", secondary_colors=["yellow"]), frozenset({"yellow"})) is None


def test_each_taste_rule_moves_the_score_and_says_why():
    request = replace(
        CASUAL,
        colors_liked=frozenset({"navy"}),
        colors_avoided=frozenset({"yellow"}),
        fit_tops=Fit.RELAXED,
        fit_bottoms=Fit.SLIM,
    )
    assert taste(bottom(name="navy", fit=Fit.SLIM), request) == (
        pytest.approx(LIKED_COLOR + PREFERRED_FIT),
        ["navy chinos is in navy, a color you like", "navy chinos has the slim fit you prefer"],
    )
    assert taste(top(name="yellow", fit=Fit.REGULAR), request) == (
        -AVOIDED_COLOR,
        ["yellow t-shirt is in yellow, a color you usually avoid"],
    )
    # the fit rule is only for tops and bottoms
    assert taste(shoes(fit=Fit.RELAXED), request) == (0.0, [])


def test_no_taste_changes_nothing():
    items = [top(), top(name="yellow"), bottom(), shoes()]
    assert suggest_outfits(items, CASUAL) == suggest_outfits(items, OutfitRequest(CASUAL.occasion))
    assert all(taste(i, CASUAL) == (0.0, []) for i in items)


def test_an_avoided_color_sinks_an_outfit_but_never_bans_it():
    white, yellow, navy = top(name="white"), top(name="yellow"), bottom()
    avoid_white = replace(CASUAL, colors_avoided=frozenset({"white"}))

    plan = suggest_outfits([white, yellow, navy], avoid_white)
    assert plan.outfits[0].pieces[0] is yellow
    assert "white t-shirt is in white, a color you usually avoid" in plan.outfits[1].reasons

    alone = suggest_outfits([white, navy], avoid_white)  # nothing else fits: still shown
    assert [p.id for p in alone.outfits[0].pieces] == [white.id, navy.id]


def test_a_liked_color_breaks_a_tie():
    sky, mint = top(name="sky"), top(name="mint")
    plan = suggest_outfits([sky, mint, bottom()], replace(CASUAL, colors_liked=frozenset({"mint"})))
    assert plan.outfits[0].pieces[0] is mint


def test_orchestrate_takes_taste_from_the_profile():
    white, yellow, navy = top(name="white"), top(name="yellow"), bottom()
    context = fake_context([white, yellow, navy], profile=Profile(colors_avoided=["Yellow"]))

    update = orchestrate_node(
        {"slots": Slots(occasion=Occasion.CASUAL, skip_weather=True)}, context=context
    )

    assert update["plan"].outfits[0].pieces[0].id == white.id


def test_avoiding_black_on_the_real_snapshot_moves_black_out_of_option_1():
    items = load_wardrobe().list_items()
    party = OutfitRequest(occasion=Occasion.PARTY, limit=3)
    assert any(
        "black" in p.primary_color_name for p in suggest_outfits(items, party).outfits[0].pieces
    )

    plan = suggest_outfits(items, replace(party, colors_avoided=frozenset({"black"})))

    assert not any("black" in p.primary_color_name for p in plan.outfits[0].pieces)
