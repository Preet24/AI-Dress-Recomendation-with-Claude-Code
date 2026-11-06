import json

import pytest
from pydantic import BaseModel

from outfit.agents.llm import LLMError, LLMRefused, chat_json, load_prompt
from outfit.config import Settings
from tests.fakes import FakeClient, reply

SETTINGS = Settings(_env_file=None, DEEPSEEK_API_KEY="test-key")
MESSAGES = [{"role": "system", "content": "Reply in JSON."}, {"role": "user", "content": "hi"}]


class Answer(BaseModel):
    value: int


def test_valid_reply_is_parsed_with_json_mode_and_agent_model():
    client = FakeClient(reply('{"value": 3}'))

    assert chat_json(client, SETTINGS, MESSAGES, Answer) == Answer(value=3)
    [request] = client.requests
    assert request["model"] == "deepseek-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == SETTINGS.agent_max_tokens


def test_invalid_reply_is_retried_once_with_the_error():
    client = FakeClient(reply('{"value": "three"}'), reply('{"value": 3}'))

    assert chat_json(client, SETTINGS, MESSAGES, Answer).value == 3
    retry = client.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "value" in retry[-1]["content"]


@pytest.mark.parametrize("first", [reply(""), reply(None), reply('{"val', "length")])
def test_empty_or_truncated_reply_is_retried(first):
    client = FakeClient(first, reply('{"value": 1}'))

    assert chat_json(client, SETTINGS, MESSAGES, Answer).value == 1


def test_second_failure_raises():
    client = FakeClient(reply("nope"), reply("still nope"))

    with pytest.raises(LLMError, match="Invalid Answer after 2 attempts"):
        chat_json(client, SETTINGS, MESSAGES, Answer)


def test_content_filter_is_not_retried():
    client = FakeClient(reply(None, "content_filter"), reply('{"value": 1}'))

    with pytest.raises(LLMRefused):
        chat_json(client, SETTINGS, MESSAGES, Answer)
    assert len(client.requests) == 1


def test_load_prompt_fills_in_the_schema():
    prompt = load_prompt("intake", Answer)

    assert "{{schema}}" not in prompt
    assert json.dumps(Answer.model_json_schema(), indent=2) in prompt
