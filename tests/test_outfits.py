import pytest

from outfit.schemas import Category, Occasion, Pattern
from outfit.tools.outfits import (
    BROKEN_RULE,
    MAX_REPEATS,
    NO_BASE,
    SINGLE_PIECE_HARMONY,
    WARM_BOTTOM,
    OutfitRequest,
    outfit_penalties,
    suggest_outfits,
)
from outfit.tools.weather import WeatherAdvice
from tests.factories import make_item

CASUAL = [Occasion.CASUAL]


def top(name="white", lab=(95.0, 0.0, 0.0), **kw):
    fields = {"category": Category.TOP, "subtype": "t-shirt", "formality": 2, "occasions": CASUAL}
    return make_item(**fields | {"primary_color_name": name, "primary_color_lab": lab} | kw)


def bottom(name="navy", lab=(20.0, 5.0, -30.0), **kw):
    fields = {"category": Category.BOTTOM, "subtype": "chinos", "formality": 2, "occasions": CASUAL}
    return make_item(**fields | {"primary_color_name": name, "primary_color_lab": lab} | kw)


def dress(**kw):
    fields = {
        "category": Category.ONE_PIECE,
        "subtype": "sundress",
        "primary_color_name": "red",
        "primary_color_lab": (53.0, 80.0, 67.0),
        "formality": 2,
        "occasions": CASUAL,
    }
    return make_item(**fields | kw)


def shoes(**kw):
    fields = {
        "category": Category.FOOTWEAR,
        "subtype": "sneakers",
        "primary_color_name": "white",
        "primary_color_lab": (95.0, 0.0, 0.0),
        "formality": 2,
        "occasions": CASUAL,
    }
    return make_item(**fields | kw)


def jacket(warmth=4, **kw):
    fields = {
        "category": Category.OUTERWEAR,
        "subtype": "jacket",
        "primary_color_name": "black",
        "primary_color_lab": (10.0, 0.0, 0.0),
        "formality": 2,
        "warmth": warmth,
        "occasions": CASUAL,
    }
    return make_item(**fields | kw)


def advice(**kw) -> WeatherAdvice:
    fields = {
        "effective_temp": 20,
        "warmth": (1, 3),
        "needs_outerwear": False,
        "outerwear_min_warmth": None,
        "avoid_flowy": False,
        "avoid_fabrics": [],
        "prefer_fabrics": [],
        "prefer_light_colors": False,
        "notes": [],
    }
    return WeatherAdvice(**fields | kw)


def ids(outfit) -> set:
    return {p.id for p in outfit.pieces}


def test_builds_top_bottom_and_one_piece_outfits_with_shoes():
    t, b, d, s = top(), bottom(), dress(), shoes()

    plan = suggest_outfits([t, b, d, s], OutfitRequest(occasion=Occasion.CASUAL))

    assert {frozenset(ids(o)) for o in plan.outfits} == {
        frozenset({t.id, b.id, s.id}),
        frozenset({d.id, s.id}),
    }
    assert plan.gaps == []
    assert all(0 <= o.score <= 1 for o in plan.outfits)


def test_base_garments_must_match_occasion_and_formality():
    office_shirt = top(formality=3, occasions=[Occasion.OFFICE])
    tee = top()  # casual, formality 2
    trousers = bottom(formality=3, occasions=[Occasion.OFFICE])

    plan = suggest_outfits([office_shirt, tee, trousers], OutfitRequest(occasion=Occasion.OFFICE))

    [outfit] = plan.outfits
    assert ids(outfit) == {office_shirt.id, trousers.id}


def test_occasion_sets_default_formality_but_request_can_override():
    assert OutfitRequest(occasion=Occasion.WEDDING).formality_range == (4, 5)
    assert OutfitRequest(occasion=Occasion.WEDDING, formality=(3, 3)).formality_range == (3, 3)
    assert OutfitRequest().formality_range == (1, 5)


def test_weather_warmth_and_wind_are_hard_rules_for_base():
    sweater = top(subtype="sweater", warmth=4)
    tee = top()
    flowy_skirt = bottom(subtype="maxi skirt", flowy=True)
    jeans = bottom(subtype="jeans")

    plan = suggest_outfits(
        [sweater, tee, flowy_skirt, jeans],
        OutfitRequest(advice=advice(warmth=(1, 2), avoid_flowy=True)),
    )

    [outfit] = plan.outfits
    assert ids(outfit) == {tee.id, jeans.id}


def test_fabrics_to_avoid_lower_the_score_but_do_not_exclude():
    satin, cotton = top(name="ivory", fabric="satin"), top(name="ivory", fabric="cotton")
    b = bottom()

    plan = suggest_outfits(
        [satin, cotton, b], OutfitRequest(advice=advice(avoid_fabrics=["satin"]))
    )

    first, second = plan.outfits
    assert cotton.id in ids(first) and satin.id in ids(second)
    assert any("satin" in r for r in second.reasons)


def test_outerwear_is_added_when_the_weather_needs_it():
    t, b, light, warm = top(), bottom(), jacket(warmth=2), jacket(warmth=4)
    cold = OutfitRequest(advice=advice(warmth=(2, 5), needs_outerwear=True, outerwear_min_warmth=4))

    plan = suggest_outfits([t, b, light, warm], cold)

    assert all(warm.id in ids(o) and light.id not in ids(o) for o in plan.outfits)
    assert suggest_outfits([t, b, light], cold).gaps == [
        "no footwear suits this request",
        "no outerwear with warmth 4+ suits this request",
    ]


def test_outerwear_is_left_out_when_not_needed():
    plan = suggest_outfits([top(), bottom(), jacket()], OutfitRequest())

    assert all(p.category is not Category.OUTERWEAR for p in plan.outfits[0].pieces)


def test_unsuitable_shoes_are_left_out_and_reported():
    heels = shoes(subtype="heels", formality=5, occasions=[Occasion.PARTY])

    plan = suggest_outfits([top(), bottom(), heels], OutfitRequest(occasion=Occasion.CASUAL))

    assert heels.id not in ids(plan.outfits[0])
    assert plan.gaps == ["no footwear suits casual"]


def test_requested_item_bypasses_rules_with_warnings_and_lower_fit():
    party_top = top(formality=4, occasions=[Occasion.PARTY])
    jeans = bottom(subtype="jeans", formality=2)
    request = OutfitRequest(occasion=Occasion.PARTY, must_include=frozenset({jeans.id}))

    [outfit] = suggest_outfits([party_top, jeans], request).outfits

    assert ids(outfit) == {party_top.id, jeans.id}
    assert len(outfit.warnings) == 2  # wrong occasion, too casual
    assert outfit.fit == pytest.approx(1 - BROKEN_RULE * 2 / 2, abs=1e-3)


def test_only_outfits_with_the_requested_item_are_returned():
    t1, t2, b = top(), top(name="grey", lab=(60.0, 0.0, 0.0)), bottom()

    plan = suggest_outfits([t1, t2, b], OutfitRequest(must_include=frozenset({t2.id})))

    assert [ids(o) for o in plan.outfits] == [{t2.id, b.id}]


def test_two_requested_tops_cannot_be_combined():
    t1, t2 = top(), top()

    plan = suggest_outfits(
        [t1, t2, bottom()], OutfitRequest(must_include=frozenset({t1.id, t2.id}))
    )

    assert plan.outfits == []
    assert "cannot be combined" in plan.gaps[-1]


def test_excluded_items_are_never_used():
    t1, t2, b = top(), top(name="grey", lab=(60.0, 0.0, 0.0)), bottom()

    plan = suggest_outfits([t1, t2, b], OutfitRequest(exclude=frozenset({t1.id})))

    assert all(t1.id not in ids(o) for o in plan.outfits)


def test_pattern_clash_and_formality_spread_are_penalized():
    plaid = top(pattern=Pattern.CHECKED, formality=1)
    floral = bottom(pattern=Pattern.FLORAL, formality=5)

    penalty, notes = outfit_penalties((plaid, floral))

    assert penalty == pytest.approx(0.15 + 0.1 * 2)
    assert any("patterned" in n for n in notes) and any("dressy" in n for n in notes)


def test_results_are_varied_but_shoes_may_repeat():
    tees = [top(name="white", lab=(95.0, 0.0, 0.0)) for _ in range(4)]
    b, s = bottom(), shoes()

    plan = suggest_outfits([*tees, b, s], OutfitRequest(limit=5))

    assert len(plan.outfits) == MAX_REPEATS  # the one bottom limits it
    assert all(s.id in ids(o) for o in plan.outfits)


def test_a_lone_one_piece_is_a_valid_outfit():
    d = dress()

    [outfit] = suggest_outfits([d], OutfitRequest()).outfits

    assert outfit.harmony == SINGLE_PIECE_HARMONY
    assert ids(outfit) == {d.id}


def test_nothing_fits_reports_a_gap():
    plan = suggest_outfits([top(), bottom()], OutfitRequest(occasion=Occasion.WEDDING))

    assert plan.outfits == []
    assert plan.gaps[-1].startswith(NO_BASE)


def test_nothing_fits_gap_names_the_closest_misses():
    tee = top(formality=2, occasions=[Occasion.PARTY])
    trousers = bottom(subtype="trousers", formality=3, warmth=4, occasions=[Occasion.PARTY])
    hot = OutfitRequest(occasion=Occasion.PARTY, advice=advice(warmth=(1, 2)))

    [gap] = [g for g in suggest_outfits([tee, trousers], hot).gaps if g.startswith(NO_BASE)]

    assert "no top fits: white t-shirt has formality 2, wanted 3-5" in gap
    assert "no bottom fits: navy trousers has warmth 4, the weather calls for 1-2" in gap
    assert "no tops in the wardrobe" in suggest_outfits([trousers], hot).gaps[-1]


def test_one_miss_lists_all_its_reasons_once_by_name():
    chinos = bottom(formality=3, occasions=[Occasion.CASUAL])
    formal = OutfitRequest(occasion=Occasion.FORMAL)

    gap = suggest_outfits([top(formality=5, occasions=[Occasion.FORMAL]), chinos], formal).gaps[-1]

    assert "no bottom fits: navy chinos is not tagged for formal and has formality 3" in gap


def test_plain_trousers_one_point_short_pass_at_formal_with_a_dressy_top():
    shirt = top(name="black", lab=(10.0, 0.0, 0.0), subtype="dress shirt", formality=5,
                occasions=[Occasion.FORMAL])  # fmt: skip
    trousers = bottom(name="black", lab=(12.0, 0.0, 0.0), subtype="trousers", formality=3,
                      occasions=[Occasion.PARTY])  # fmt: skip
    formal = OutfitRequest(occasion=Occasion.FORMAL)

    [outfit] = suggest_outfits([shirt, trousers], formal).outfits

    assert ids(outfit) == {shirt.id, trousers.id}
    assert "black trousers is a little less dressy, so the top sets the tone" in outfit.reasons
    assert outfit.fit < 1.0  # penalized
    # Never for casual bottoms, patterns, two points short, or a less dressy range
    for other in (
        bottom(subtype="chinos", formality=3, occasions=[Occasion.PARTY]),
        bottom(subtype="trousers", formality=3, fabric="denim", occasions=[Occasion.PARTY]),
        bottom(
            subtype="trousers", formality=3, pattern=Pattern.CHECKED, occasions=[Occasion.PARTY]
        ),  # fmt: skip
        bottom(subtype="trousers", formality=2, occasions=[Occasion.PARTY]),
    ):
        assert suggest_outfits([shirt, other], formal).outfits == []
    office = OutfitRequest(occasion=Occasion.OFFICE)
    low = bottom(subtype="trousers", formality=2, occasions=[Occasion.OFFICE])
    assert (
        suggest_outfits([top(formality=3, occasions=[Occasion.OFFICE]), low], office).outfits == []
    )


def test_lounge_needs_lounge_bottoms_and_adds_no_shoes():
    tee = top(formality=1, occasions=[Occasion.CASUAL])  # a casual tee passes as near
    jeans = bottom(subtype="jeans", formality=2, occasions=[Occasion.CASUAL])
    pyjamas = bottom(subtype="pyjama pants", formality=1, occasions=[Occasion.LOUNGE])
    lounge = OutfitRequest(occasion=Occasion.LOUNGE)

    plan = suggest_outfits([tee, jeans, pyjamas, shoes()], lounge)

    assert [ids(o) for o in plan.outfits] == [{tee.id, pyjamas.id}]
    assert plan.gaps == []  # no "no footwear" either
    assert suggest_outfits([tee, jeans], lounge).outfits == []


def test_a_garment_tagged_for_a_near_occasion_passes_when_dressy_enough():
    office_shirt = top(name="white", formality=3, occasions=[Occasion.OFFICE])
    party_shirt = top(name="grey", lab=(60.0, 0.0, 0.0), formality=3, occasions=[Occasion.PARTY])
    casual_tee = top(name="black", lab=(10.0, 0.0, 0.0), formality=2)  # casual, too relaxed
    chinos = bottom(formality=3)  # tagged casual only
    request = OutfitRequest(occasion=Occasion.PARTY, limit=5)

    plan = suggest_outfits([office_shirt, party_shirt, casual_tee, chinos], request)

    assert [ids(o) for o in plan.outfits] == [
        {party_shirt.id, chinos.id},  # one near piece beats two
        {office_shirt.id, chinos.id},
    ]
    assert any("not tagged for party, but is dressy enough" in r for r in plan.outfits[0].reasons)


def test_near_occasions_do_not_cross_into_unrelated_ones():
    gym_top = top(formality=3, occasions=[Occasion.SPORTS])

    plan = suggest_outfits([gym_top, bottom(formality=3)], OutfitRequest(occasion=Occasion.OFFICE))

    assert plan.outfits == []


def test_bottoms_may_be_one_step_warmer_than_the_weather_but_tops_may_not():
    tee, warm_top = top(warmth=2), top(name="grey", lab=(60.0, 0.0, 0.0), warmth=3)
    jeans = bottom(subtype="jeans", warmth=3)
    wool_trousers = bottom(subtype="trousers", warmth=4)

    plan = suggest_outfits(
        [tee, warm_top, jeans, wool_trousers], OutfitRequest(advice=advice(warmth=(1, 2)))
    )

    [outfit] = plan.outfits
    assert ids(outfit) == {tee.id, jeans.id}
    assert "navy jeans is a bit warm for this weather" in outfit.reasons
    assert outfit.fit == pytest.approx((1 + 1 - WARM_BOTTOM) / 2, abs=1e-3)


def test_better_colors_rank_higher():
    white_tee = top()
    red, green = (
        bottom(name="red", lab=(53.0, 80.0, 67.0)),
        bottom(name="olive", lab=(52.0, -10.0, 45.0)),
    )
    red_top = top(name="red", lab=(53.0, 80.0, 67.0))

    plan = suggest_outfits([white_tee, red_top, red, green], OutfitRequest(limit=4))

    best = plan.outfits[0]
    assert white_tee.id in ids(best)  # neutral base beats two competing colors
