"""Look scores: one DeepSeek vision call about the outfit in a wear-log photo, never the person."""

import base64
import json

import pytest

from outfit.agents.llm import LLMError
from outfit.colouring import SEND_MAX_SIDE
from outfit.look import METRICS, NoPersonVisible, score_look
from outfit.profile import Colouring
from outfit.schemas import Occasion
from tests.fakes import FakeClient, reply
from tests.test_packing import top, trousers
from tests.test_profile_photo import SETTINGS, opened, photo_with_exif

PIECES = [top(), trousers()]
COLOURING = Colouring(summary="warm undertone, medium depth")
LOOK = {
    "person_visible": True,
    **{m: {"score": 4, "reason": f"{m} is good"} for m in METRICS},
    "tips": ["Tuck the shirt in", "Try brown shoes", "A third tip is dropped"],
}


def score(client, colouring=COLOURING, occasion=Occasion.OFFICE):
    return score_look(photo_with_exif(), PIECES, occasion, colouring, client, SETTINGS)


def test_one_call_with_a_small_copy_and_what_was_worn():
    client = FakeClient(reply(json.dumps(LOOK)))

    look = score(client)

    assert look.fit.score == 4 and look.polish.reason == "polish is good"
    assert look.tips == ["Tuck the shirt in", "Try brown shoes"]
    [request] = client.requests
    assert request["model"] == SETTINGS.tagger_model
    system, user = request["messages"]
    assert "never the person" in system["content"]
    assert "body, shape, weight" in system["content"]
    image, text = user["content"]
    sent = base64.standard_b64decode(image["image_url"]["url"].split(",", 1)[1])
    assert max(opened(sent).size) == SEND_MAX_SIDE and not opened(sent).getexif()
    assert "white oxford shirt, navy trousers" in text["text"]
    assert "Occasion: office." in text["text"]
    assert "Their colouring: warm undertone, medium depth." in text["text"]


def test_without_colouring_or_occasion_the_model_is_told_so():
    client = FakeClient(reply(json.dumps(LOOK)))
    score(client, colouring=None, occasion=None)
    text = client.requests[0]["messages"][1]["content"][1]["text"]
    assert "Occasion: not given." in text and "Their colouring: unknown." in text


def test_no_person_is_an_error_not_a_guess():
    client = FakeClient(reply(json.dumps({"person_visible": False})))
    with pytest.raises(NoPersonVisible):
        score(client)


def test_a_missing_score_is_retried_then_given_up():
    partial = {k: v for k, v in LOOK.items() if k != "colour"}
    client = FakeClient(reply(json.dumps(partial)), reply(json.dumps(partial)))
    with pytest.raises(LLMError, match="colour"):
        score(client)
    assert len(client.requests) == 2


def test_scores_stay_within_1_to_5():
    bad = LOOK | {"fit": {"score": 7, "reason": "great"}}
    client = FakeClient(reply(json.dumps(bad)), reply(json.dumps(LOOK)))
    assert score(client).fit.score == 4 and len(client.requests) == 2
