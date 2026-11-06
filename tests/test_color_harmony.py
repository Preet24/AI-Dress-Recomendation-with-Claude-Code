import pytest

from outfit.tools.color_harmony import (
    NEUTRAL_PAIR,
    SOFT_PAIR,
    Relation,
    Swatch,
    hue_distance,
    hue_relation,
    hue_score,
    outfit_harmony,
    pair_harmony,
    ryb_hue,
    swatch,
)
from tests.factories import make_item

# Textbook sRGB primaries/secondaries in CIELAB.
LAB = {
    "red": (53.2, 80.1, 67.2),
    "orange": (67.0, 43.0, 74.0),
    "yellow": (97.1, -21.6, 94.5),
    "green": (46.2, -51.7, 49.9),
    "blue": (32.3, 79.2, -107.9),
    "purple": (30.0, 58.0, -36.0),
}


def colored(name: str, hue: float, chroma: float = 60, lightness: float = 50) -> Swatch:
    return Swatch(name, lightness, chroma, hue, neutral=False)


def neutral(name: str, lightness: float) -> Swatch:
    return Swatch(name, lightness, 2.0, 0.0, neutral=True)


@pytest.mark.parametrize(
    "name, hue",
    [("red", 0), ("orange", 60), ("yellow", 120), ("green", 180), ("blue", 240), ("purple", 300)],
)
def test_hue_is_measured_on_the_ryb_wheel(name, hue):
    assert ryb_hue(LAB[name]) == pytest.approx(hue, abs=8) or (
        hue == 0 and ryb_hue(LAB[name]) > 352
    )


@pytest.mark.parametrize("a, b", [("red", "green"), ("blue", "orange"), ("yellow", "purple")])
def test_styling_complements_are_opposite(a, b):
    assert hue_distance(ryb_hue(LAB[a]), ryb_hue(LAB[b])) == pytest.approx(180, abs=10)


def test_hue_distance_wraps_around():
    assert hue_distance(350, 10) == 20
    assert hue_distance(0, 180) == 180


@pytest.mark.parametrize(
    "distance, relation",
    [(10, "analogous"), (60, "awkward hue gap"), (120, "triadic"), (175, "complementary")],
)
def test_hue_relation(distance, relation):
    assert hue_relation(distance) == relation


def test_hue_score_favors_close_or_opposite_hues():
    assert hue_score(15) > hue_score(90)
    assert hue_score(180) > hue_score(90)
    assert hue_score(90) == min(hue_score(d) for d in range(0, 181, 5))


@pytest.mark.parametrize(
    "color_name, lab, fabric, expected",
    [
        ("olive grey", (59, 0.9, 6.4), "cotton", True),  # named neutral
        ("off-white", (80, 7.3, -7.2), "cotton", True),
        ("medium blue", (49, -1.7, -8.0), "denim", True),  # denim acts neutral
        ("slate", (50, 1.0, 2.0), "wool", True),  # unnamed but colorless
        ("lavender", (82, 4.5, -7.9), "cotton", False),  # pastel: low chroma, still a color
        ("mauve", (67, 15.2, 11.6), "cotton", False),
    ],
)
def test_swatch_neutrality_uses_name_fabric_then_chroma(color_name, lab, fabric, expected):
    item = make_item(primary_color_name=color_name, primary_color_lab=lab, fabric=fabric)

    assert swatch(item).neutral is expected


def test_neutral_base_beats_two_neutrals_and_awkward_colors():
    black, white = neutral("black", 5), neutral("white", 95)
    red, orange_ish = colored("red", 0), colored("amber", 85)

    accent = pair_harmony(black, red)
    neutrals = pair_harmony(black, white)
    awkward = pair_harmony(red, orange_ish)

    assert accent.relation is Relation.NEUTRAL_BASE and "black" in accent.reason
    assert neutrals.relation is Relation.NEUTRALS
    assert awkward.relation is Relation.AWKWARD
    assert accent.score > awkward.score
    assert neutrals.score > awkward.score


def test_complementary_and_analogous_beat_awkward():
    red = colored("red", 0)

    assert pair_harmony(red, colored("green", 180)).relation is Relation.COMPLEMENTARY
    assert pair_harmony(red, colored("coral", 20, lightness=80)).relation is Relation.ANALOGOUS
    awkward = pair_harmony(red, colored("amber", 85)).score
    assert pair_harmony(red, colored("green", 180)).score > awkward
    assert pair_harmony(red, colored("coral", 20)).score > awkward


def test_washed_out_colors_ignore_their_noisy_hue():
    # Two pastels 90 degrees apart: with real photo chroma, judged as soft colors.
    a, b = colored("blush", 0, chroma=8, lightness=80), colored("mint", 90, chroma=8, lightness=60)

    result = pair_harmony(a, b)

    assert result.relation is Relation.SOFT
    assert result.score == pytest.approx(SOFT_PAIR + 0.05, abs=0.03)  # + some contrast


def test_two_loud_clashing_colors_are_penalized():
    loud = pair_harmony(colored("red", 0, 70), colored("lime", 150, 70, 80))
    calm = pair_harmony(colored("red", 0, 35), colored("lime", 150, 35, 80))

    assert "both loud" in loud.reason
    assert loud.score < calm.score


def test_light_dark_contrast_helps():
    flat = pair_harmony(neutral("grey", 50), neutral("grey", 52))
    contrast = pair_harmony(neutral("black", 10), neutral("white", 90))

    assert flat.score < NEUTRAL_PAIR < contrast.score
    assert "little contrast" in flat.reason and "good light-dark contrast" in contrast.reason


def test_scores_stay_within_0_and_1():
    for a in (colored("x", h, c, lum) for h in (0, 90, 180) for c in (5, 80) for lum in (5, 95)):
        for b in (neutral("n", 50), colored("y", 45, 90, 50)):
            assert 0 <= pair_harmony(a, b).score <= 1


def item(name: str, lab: tuple[float, float, float], fabric: str = "cotton"):
    return make_item(primary_color_name=name, primary_color_lab=lab, fabric=fabric)


def test_outfit_harmony_averages_pairs():
    outfit = [
        item("black", (5, 0, 0)),
        item("red", LAB["red"]),
        item("medium blue", (49, -2, -8), "denim"),
    ]

    result = outfit_harmony(outfit)

    assert len(result.pairs) == 3
    assert result.colors == ["red"]
    assert result.reason == "neutrals with a red accent"
    assert result.score == pytest.approx(sum(p.score for p in result.pairs) / 3, abs=1e-3)


def test_more_than_two_colors_is_penalized():
    three = [item("red", LAB["red"]), item("yellow", LAB["yellow"]), item("blue", LAB["blue"])]

    result = outfit_harmony(three)

    plain_mean = sum(p.score for p in result.pairs) / len(result.pairs)
    assert result.score == pytest.approx(plain_mean - 0.1, abs=1e-3)
    assert "busy" in result.reason


def test_outfit_needs_two_pieces():
    with pytest.raises(ValueError, match="two pieces"):
        outfit_harmony([item("black", (5, 0, 0))])
