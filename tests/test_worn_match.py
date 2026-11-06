"""Matching a worn photo's garments with the wardrobe and the logged outfit (pure)."""

import numpy as np
import pytest

from outfit.schemas import Category
from outfit.tools.worn_match import CLEAR_WIN, NOT_IN_WARDROBE, match_garment, match_photo
from outfit.wear_log import Verdict
from tests.factories import make_item

SHIRT = make_item(subtype="shirt", primary_color_name="black")
POLO = make_item(subtype="polo", primary_color_name="turquoise")
CHINOS = make_item(category=Category.BOTTOM, subtype="chinos", primary_color_name="tan")
TROUSERS = make_item(category=Category.BOTTOM, subtype="trousers", primary_color_name="black")
SHOES = make_item(category=Category.FOOTWEAR, subtype="loafers")
WARDROBE = [SHIRT, POLO, CHINOS, TROUSERS, SHOES]
# One axis per piece: a photo vector's component i is its similarity to piece i.
VECTORS = np.eye(len(WARDROBE), dtype=np.float32)


def photo(**scores: float) -> np.ndarray:
    names = ["shirt", "polo", "chinos", "trousers", "shoes"]
    return np.array([scores.get(n, 0.0) for n in names], dtype=np.float32)


def match(region, vector, *logged):
    return match_garment(region, vector, WARDROBE, VECTORS, {i.id for i in logged})


def test_the_logged_piece_as_best_match_is_yours():
    found = match("upper", photo(shirt=0.77, polo=0.66), SHIRT, CHINOS)
    assert (found.verdict, found.logged, found.best) == (Verdict.YOURS, SHIRT.id, SHIRT.id)
    assert found.message == "your black shirt: yes"


def test_a_close_runner_up_still_counts_as_the_logged_piece():
    # Dark trousers vs chinos were 0.01 apart on real photos: not enough to doubt.
    found = match("lower", photo(trousers=0.80, chinos=0.79), CHINOS)
    assert found.verdict is Verdict.YOURS and found.logged == CHINOS.id
    assert (found.score, found.logged_score) == (0.8, 0.79)


def test_a_clear_winner_asks_which_it_was():
    found = match("lower", photo(trousers=0.82, chinos=0.82 - CLEAR_WIN - 0.01), CHINOS)
    assert found.verdict is Verdict.UNSURE and found.best == TROUSERS.id
    assert found.message == (
        "this bottom looks more like your black trousers than your tan chinos: which was it?"
    )


def test_a_garment_not_logged_is_extra():
    found = match("upper", photo(polo=0.82), CHINOS)
    assert found.verdict is Verdict.EXTRA and found.logged is None and found.best == POLO.id


@pytest.mark.parametrize("logged", [(), (SHIRT,)])
def test_a_weak_best_match_may_not_be_in_the_wardrobe(logged):
    found = match("upper", photo(shirt=NOT_IN_WARDROBE - 0.01), *logged)
    assert found.verdict is Verdict.UNKNOWN
    says = "or anything else in your wardrobe" if logged else "may not be in your wardrobe"
    assert found.message.endswith(says)


def test_only_pieces_of_the_regions_kind_are_compared():
    found = match("upper", photo(trousers=0.95, shirt=0.75), SHIRT)  # trousers: not a top
    assert found.best == SHIRT.id and found.verdict is Verdict.YOURS


def test_a_one_piece_with_none_in_the_wardrobe():
    found = match("full", photo(shirt=0.9))
    assert (found.verdict, found.best, found.score) == (Verdict.UNKNOWN, None, 0.0)


def test_the_photo_report_lists_pieces_it_cannot_show():
    report = match_photo(
        [("upper", photo(shirt=0.77)), ("lower", photo(chinos=0.82))],
        WARDROBE,
        VECTORS,
        [SHIRT, CHINOS, SHOES],
    )
    assert [g.verdict for g in report.garments] == [Verdict.YOURS, Verdict.YOURS]
    assert report.unchecked == [SHOES.id]  # shoes aren't cut out of a worn photo


def test_an_empty_wardrobe_finds_nothing():
    report = match_photo([("upper", photo(shirt=1.0))], [], np.zeros((0, 0)), [])
    assert report.garments[0].verdict is Verdict.UNKNOWN
