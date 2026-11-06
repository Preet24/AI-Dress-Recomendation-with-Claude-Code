import pytest

from outfit.agents.orchestrate import orchestrate_node
from outfit.agents.state import Slots
from outfit.schemas import Category, Occasion, Pattern
from outfit.tools.accessories import (
    Kind,
    choose_accessories,
    finish_plan,
    is_shirt,
    kind_of,
    takes_belt,
)
from outfit.tools.outfits import OutfitRequest, suggest_outfits
from tests.factories import make_item
from tests.fakes import fake_context
from tests.test_outfits import advice

OFFICE = OutfitRequest(occasion=Occasion.OFFICE)
CASUAL = OutfitRequest(occasion=Occasion.CASUAL)
BOTH = [Occasion.OFFICE, Occasion.CASUAL, Occasion.FORMAL, Occasion.PARTY, Occasion.WEDDING]


def accessory(subtype, color="black", lab=(10.0, 0.0, 0.0), **kw):
    fields = {
        "category": Category.ACCESSORY,
        "subtype": subtype,
        "primary_color_name": color,
        "primary_color_lab": lab,
        "secondary_colors": [],
        "fabric": "leather",
        "formality": 3,
        "warmth": 1,
        "occasions": BOTH,
    }
    return make_item(**fields | kw)


def shirt(**kw):
    return make_item(**{"subtype": "oxford shirt", "primary_color_name": "white",
                        "primary_color_lab": (95.0, 0.0, 0.0), "occasions": BOTH} | kw)  # fmt: skip


def trousers(**kw):
    fields = {
        "category": Category.BOTTOM,
        "subtype": "trousers",
        "primary_color_name": "navy",
        "primary_color_lab": (20.0, 5.0, -30.0),
        "occasions": BOTH,
    }
    return make_item(**fields | kw)


def shoes(color="black", lab=(10.0, 0.0, 0.0)):
    return make_item(category=Category.FOOTWEAR, subtype="oxford shoes", primary_color_name=color,
                     primary_color_lab=lab, occasions=BOTH)  # fmt: skip


BROWN = ("brown", (35.0, 12.0, 25.0))


def chosen(finish) -> dict[Kind, str]:
    return {c.kind: c.item.subtype for c in finish.chosen}


@pytest.mark.parametrize(
    ("subtype", "kind"),
    [
        ("leather belt", Kind.BELT),
        ("silk tie", Kind.TIE),
        ("bow tie", Kind.TIE),
        ("belt bag", Kind.BAG),  # the last kind word wins
        ("sling-bag", Kind.BAG),
        ("clutch", Kind.BAG),
        ("aviator sunglasses", Kind.SUNGLASSES),
        ("baseball cap", Kind.HAT),
        ("wool scarf", Kind.SCARF),
        ("dupatta", Kind.DUPATTA),
        ("analog watch", Kind.WATCH),
        ("jhumkas", Kind.JEWELLERY),
        ("keychain", None),  # whole words only
    ],
)
def test_kind_comes_from_the_subtype(subtype, kind):
    assert kind_of(accessory(subtype)) is kind


def test_only_accessories_have_a_kind():
    assert kind_of(make_item(category=Category.TOP, subtype="tie-dye t-shirt")) is None


@pytest.mark.parametrize(
    ("subtype", "belted"),
    [("trousers", True), ("slim jeans", True), ("chino shorts", True), ("track pants", False),
     ("leggings", False)],
)  # fmt: skip
def test_belts_go_with_trousers_jeans_and_chinos(subtype, belted):
    assert takes_belt(trousers(subtype=subtype)) is belted


@pytest.mark.parametrize(
    ("subtype", "is_one"),
    [("oxford shirt", True), ("t-shirt", False), ("polo shirt", False), ("sweatshirt", False)],
)
def test_ties_go_with_shirts_only(subtype, is_one):
    assert is_shirt(shirt(subtype=subtype)) is is_one


def test_belt_needs_a_belted_bottom():
    belt = accessory("leather belt")

    with_trousers = choose_accessories([belt], (shirt(), trousers()), OFFICE)
    with_track_pants = choose_accessories(
        [belt], (shirt(), trousers(subtype="track pants")), OFFICE
    )

    assert chosen(with_trousers) == {Kind.BELT: "leather belt"}
    assert with_track_pants.chosen == []


def test_belt_matches_the_shoes_leather():
    black, brown = accessory("black belt"), accessory("brown belt", *BROWN)
    outfit = (shirt(), trousers())

    assert chosen(choose_accessories([brown, black], (*outfit, shoes()), OFFICE)) == {
        Kind.BELT: "black belt"
    }
    finish = choose_accessories([black, brown], (*outfit, shoes(*BROWN)), OFFICE)
    assert chosen(finish) == {Kind.BELT: "brown belt"}
    assert any("matches" in r for r in finish.chosen[0].reasons)


def test_tie_needs_a_shirt_and_a_dressy_occasion():
    tie = accessory("silk tie", "maroon", (30.0, 35.0, 12.0), fabric="silk", formality=4)

    assert chosen(choose_accessories([tie], (shirt(), trousers()), OFFICE)) == {
        Kind.TIE: "silk tie"
    }
    assert choose_accessories([tie], (shirt(), trousers()), CASUAL).chosen == []
    assert choose_accessories([tie], (shirt(subtype="t-shirt"), trousers()), OFFICE).chosen == []


def test_patterned_tie_loses_to_a_plain_one_on_a_patterned_shirt():
    plain = accessory("plain tie", "navy", (20.0, 5.0, -30.0), formality=4)
    striped = accessory("striped tie", "navy", (20.0, 5.0, -30.0), formality=4,
                        pattern=Pattern.STRIPED)  # fmt: skip
    checked = shirt(pattern=Pattern.CHECKED)

    finish = choose_accessories([striped, plain], (checked, trousers()), OFFICE)

    assert chosen(finish) == {Kind.TIE: "plain tie"}


def test_a_tie_is_missed_only_at_formal_occasions_by_people_who_own_ties():
    novelty = accessory("novelty tie", "yellow", (85.0, 0.0, 70.0), formality=1,
                        occasions=[Occasion.CASUAL])  # fmt: skip
    formal = OutfitRequest(occasion=Occasion.FORMAL)
    outfit = (shirt(formality=4), trousers(formality=4))

    assert choose_accessories([novelty], outfit, formal).gaps == ["no tie suits formal"]
    assert choose_accessories([], outfit, formal).gaps == []
    assert choose_accessories([novelty], outfit, OFFICE).gaps == []


def test_sunglasses_only_in_strong_sun_or_on_a_ride():
    glasses = accessory("sunglasses", formality=2)
    outfit = (shirt(), trousers())

    def with_weather(**kw):
        return choose_accessories([glasses], outfit, OutfitRequest(advice=advice(**kw)))

    assert chosen(with_weather(sunny=True)) == {Kind.SUNGLASSES: "sunglasses"}
    assert chosen(with_weather(riding=True)) == {Kind.SUNGLASSES: "sunglasses"}
    assert with_weather().chosen == []
    assert choose_accessories([glasses], outfit, OFFICE).chosen == []  # no forecast


def test_strong_sun_without_sunglasses_is_a_gap():
    sunny = OutfitRequest(advice=advice(sunny=True))
    riding = OutfitRequest(advice=advice(riding=True))

    assert choose_accessories([], (shirt(), trousers()), sunny).gaps == [
        "no sunglasses for the strong sun"
    ]
    assert choose_accessories([], (shirt(), trousers()), riding).gaps == []


def test_at_home_only_a_requested_accessory_and_nothing_missing():
    belt = accessory("leather belt", occasions=[Occasion.LOUNGE, Occasion.CASUAL])
    outfit = (shirt(), trousers())
    lounge = OutfitRequest(occasion=Occasion.LOUNGE, advice=advice(sunny=True))

    assert choose_accessories([belt], outfit, lounge).chosen == []
    assert choose_accessories([belt], outfit, lounge).gaps == []  # no sunglasses gap
    asked = OutfitRequest(occasion=Occasion.LOUNGE, must_include=frozenset({belt.id}))
    assert chosen(choose_accessories([belt], outfit, asked)) == {Kind.BELT: "leather belt"}


def test_scarf_in_the_cold_and_no_warm_hat_in_the_heat():
    scarf = accessory("wool scarf", "grey", (55.0, 0.0, 0.0), fabric="wool", warmth=4)
    beanie = accessory("beanie", "grey", (55.0, 0.0, 0.0), fabric="wool", warmth=4)
    cap = accessory("cap", "white", (95.0, 0.0, 0.0), fabric="cotton", formality=1)
    outfit = (shirt(), trousers())
    cold = OutfitRequest(advice=advice(effective_temp=5, warmth=(2, 5)))
    hot_sun = OutfitRequest(advice=advice(effective_temp=32, warmth=(1, 2), sunny=True))

    assert chosen(choose_accessories([scarf], outfit, cold)) == {Kind.SCARF: "wool scarf"}
    assert choose_accessories([], outfit, cold).gaps == ["no scarf for the cold"]
    assert choose_accessories([scarf], outfit, OutfitRequest(advice=advice())).chosen == []
    assert chosen(choose_accessories([beanie, cap], outfit, hot_sun)) == {Kind.HAT: "cap"}


def test_dupatta_only_with_ethnic_wear_and_never_flowy_on_a_ride():
    dupatta = accessory("dupatta", "gold", (70.0, 5.0, 45.0), fabric="silk", flowy=True)
    kurta = make_item(category=Category.ETHNIC_TOP, subtype="kurta", occasions=BOTH)
    churidar = make_item(category=Category.ETHNIC_BOTTOM, subtype="churidar", occasions=BOTH)

    assert chosen(choose_accessories([dupatta], (kurta, churidar), OFFICE)) == {
        Kind.DUPATTA: "dupatta"
    }
    assert choose_accessories([dupatta], (shirt(), trousers()), OFFICE).chosen == []
    riding = OutfitRequest(advice=advice(riding=True, avoid_flowy=True))
    assert choose_accessories([dupatta], (kurta, churidar), riding).chosen == []


def test_one_per_kind_the_dressier_bag_for_a_party():
    clutch = accessory("clutch", formality=4, occasions=[Occasion.PARTY])
    backpack = accessory("backpack", fabric="nylon", formality=1, occasions=[Occasion.CASUAL])
    watch = accessory("analog watch", formality=3)
    party = OutfitRequest(occasion=Occasion.PARTY)

    finish = choose_accessories([backpack, clutch, watch], (shirt(), trousers()), party)

    assert chosen(finish) == {Kind.BAG: "clutch", Kind.WATCH: "analog watch"}


def test_a_requested_accessory_is_kept_with_warnings():
    tie = accessory("silk tie", formality=4)
    request = OutfitRequest(occasion=Occasion.CASUAL, must_include=frozenset({tie.id}))

    finish = choose_accessories([tie], (shirt(subtype="t-shirt"), trousers()), request)

    assert chosen(finish) == {Kind.TIE: "silk tie"}
    assert {"black silk tie is too dressy for casual", "black silk tie needs a shirt"} == set(
        finish.chosen[0].warnings
    )


def test_excluded_accessories_are_skipped():
    belt = accessory("leather belt")
    request = OutfitRequest(occasion=Occasion.OFFICE, exclude=frozenset({belt.id}))

    assert choose_accessories([belt], (shirt(), trousers()), request).chosen == []


def test_finish_plan_adds_accessories_after_the_pieces_and_keeps_the_score():
    belt, watch = accessory("leather belt"), accessory("analog watch")
    items = [shirt(), trousers(), shoes(), belt, watch]
    plan = suggest_outfits(items, OFFICE)

    finished = finish_plan(plan, items, OFFICE)

    best = finished.outfits[0]
    assert [p.subtype for p in best.pieces][-2:] == ["leather belt", "analog watch"]
    assert best.pieces[:-2] == plan.outfits[0].pieces
    assert best.score == plan.outfits[0].score
    assert any("matches the black oxford shoes" in r for r in best.reasons)


def test_finish_plan_lists_each_accessory_gap_once():
    items = [shirt(), shirt(primary_color_name="blue"), trousers()]
    sunny = OutfitRequest(occasion=Occasion.OFFICE, advice=advice(sunny=True))

    finished = finish_plan(suggest_outfits(items, sunny), items, sunny)

    assert len(finished.outfits) == 2
    assert finished.gaps.count("no sunglasses for the strong sun") == 1


def test_a_requested_accessory_does_not_block_the_outfits():
    tie = accessory("silk tie", formality=4)
    items = [shirt(), trousers(), tie]
    request = OutfitRequest(occasion=Occasion.OFFICE, must_include=frozenset({tie.id}))

    finished = finish_plan(suggest_outfits(items, request), items, request)

    assert finished.outfits and all(tie in o.pieces for o in finished.outfits)
    assert "the requested items cannot be combined into one outfit" not in finished.gaps


def test_orchestrate_plans_with_accessories():
    belt = accessory("leather belt")
    context = fake_context([shirt(), trousers(), belt])

    update = orchestrate_node({"slots": Slots(occasion=Occasion.OFFICE)}, context=context)

    assert belt in update["plan"].outfits[0].pieces


def test_sunglasses_are_not_judged_by_the_occasion():
    casual_glasses = accessory("sunglasses", formality=1, occasions=[Occasion.CASUAL])
    party = OutfitRequest(occasion=Occasion.PARTY, advice=advice(sunny=True))

    finish = choose_accessories([casual_glasses], (shirt(), trousers()), party)

    assert chosen(finish) == {Kind.SUNGLASSES: "sunglasses"}
    assert not any("casual" in r or "not really for" in r for r in finish.chosen[0].reasons)


def test_other_accessories_still_are():
    cap = accessory("baseball cap", formality=1, occasions=[Occasion.SPORTS])
    wedding = OutfitRequest(occasion=Occasion.WEDDING, advice=advice(sunny=True))

    assert (
        choose_accessories([cap], (shirt(formality=5), trousers(formality=5)), wedding).chosen == []
    )
