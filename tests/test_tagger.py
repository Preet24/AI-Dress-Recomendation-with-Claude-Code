import base64
import io
import json

import pytest
from PIL import Image

from outfit.config import Settings
from outfit.ingestion.embedder import BACKGROUND_RGB
from outfit.ingestion.tagger import (
    TAG_SCHEMA,
    TaggingError,
    TagResult,
    encode_png,
    load_prompt,
    make_client,
    tag_garment,
)
from outfit.schemas import Category, Fit, Length, Occasion, Pattern
from tests.fakes import VALID_TAGS, FakeClient, reply


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")


@pytest.fixture
def cutout() -> Image.Image:
    img = Image.new("RGBA", (8, 8), (0, 0, 255, 0))  # transparent, blue underneath
    img.putpixel((4, 4), (200, 20, 30, 255))
    return img


def test_valid_reply_becomes_tag_result(settings, cutout):
    client = FakeClient(reply(json.dumps(VALID_TAGS)))

    tags = tag_garment(cutout, "whole", client, settings)

    assert tags.category is Category.TOP
    assert tags.occasions == [Occasion.CASUAL, Occasion.TRAVEL]
    assert len(client.requests) == 1


def test_request_uses_deepseek_model_json_mode_and_prompt(settings, cutout):
    client = FakeClient(reply(json.dumps(VALID_TAGS)))

    tag_garment(cutout, "upper", client, settings)

    [request] = client.requests
    assert request["model"] == "deepseek-flash"
    assert request["response_format"] == {"type": "json_object"}
    system, user = request["messages"]
    assert system == {"role": "system", "content": load_prompt()}
    text_block, image_block = user["content"]
    assert "`upper`" in text_block["text"]
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_owner_note_is_sent_with_the_image(settings, cutout):
    client = FakeClient(reply(json.dumps(VALID_TAGS)))

    tag_garment(cutout, "upper", client, settings, note="a midi slip dress")

    text_block, _ = client.requests[0]["messages"][1]["content"]
    assert "Owner's note: a midi slip dress" in text_block["text"]


def test_prompt_spells_out_json_schema_with_every_enum_value():
    prompt = load_prompt()

    assert "{{schema}}" not in prompt
    assert "json" in prompt.lower()  # DeepSeek JSON mode requires the word in the prompt
    for enum in (Category, Pattern, Length, Fit, Occasion):
        assert all(f'"{v.value}"' in prompt for v in enum)


def test_invalid_reply_is_retried_once_with_the_error(settings, cutout):
    bad = {**VALID_TAGS, "formality": 9}
    client = FakeClient(reply(json.dumps(bad)), reply(json.dumps(VALID_TAGS)))

    tags = tag_garment(cutout, "whole", client, settings)

    assert tags.formality == 2
    retry = client.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2]["content"] == json.dumps(bad)
    assert "formality" in retry[3]["content"]


def test_second_invalid_reply_raises(settings, cutout):
    client = FakeClient(reply("not json"), reply(json.dumps({**VALID_TAGS, "occasions": []})))

    with pytest.raises(TaggingError, match="after 2 attempts"):
        tag_garment(cutout, "whole", client, settings)


@pytest.mark.parametrize(
    "first",
    [reply('{"category": "to', "length"), reply(""), reply(None)],
    ids=["truncated", "empty", "none"],
)
def test_truncated_or_empty_reply_is_retried(settings, cutout, first):
    client = FakeClient(first, reply(json.dumps(VALID_TAGS)))

    assert tag_garment(cutout, "whole", client, settings).subtype == "t-shirt"
    assert len(client.requests) == 2


def test_content_filter_raises_without_retry(settings, cutout):
    client = FakeClient(reply(None, "content_filter"), reply(json.dumps(VALID_TAGS)))

    with pytest.raises(TaggingError, match="declined"):
        tag_garment(cutout, "whole", client, settings)
    assert len(client.requests) == 1


def test_cutout_is_sent_on_grey_not_its_old_background(cutout):
    png = base64.standard_b64decode(encode_png(cutout))

    sent = Image.open(io.BytesIO(png))

    assert sent.mode == "RGB"
    assert sent.getpixel((0, 0)) == BACKGROUND_RGB
    assert sent.getpixel((4, 4)) == (200, 20, 30)


def test_schema_matches_tag_result_fields():
    assert set(TAG_SCHEMA["properties"]) == set(TagResult.model_fields)
    assert TAG_SCHEMA["required"] == list(TagResult.model_fields)


def test_client_points_at_deepseek(settings):
    client = make_client(settings)

    assert str(client.base_url).startswith("https://api.deepseek.com")


def test_missing_api_key_gives_clear_error():
    with pytest.raises(TaggingError, match="DEEPSEEK_API_KEY"):
        make_client(Settings(_env_file=None, DEEPSEEK_API_KEY=None))
