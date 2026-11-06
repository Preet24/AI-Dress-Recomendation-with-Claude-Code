"""Structured DeepSeek calls for the agents: JSON mode + Pydantic + one retry.

The same contract as the tagger: JSON mode guarantees a JSON object but not our
schema, so the schema is spelled out in the prompt (`{{schema}}`) and Pydantic
validates. Invalid, empty or truncated output gets one retry with the error fed
back; `content_filter` is not retried.
"""

import json
import logging
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from outfit.config import Settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
SCHEMA_PLACEHOLDER = "{{schema}}"
MAX_ATTEMPTS = 2

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model did not return a usable reply."""


class LLMRefused(LLMError):
    """The model declined (content_filter); retrying will not help."""


@cache
def load_prompt(name: str, schema: type[BaseModel] | None = None) -> str:
    """System prompt `prompts/<name>.md`, with the reply schema filled in if given."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if schema is not None:
        text = text.replace(SCHEMA_PLACEHOLDER, json.dumps(schema.model_json_schema(), indent=2))
    return text


def parse_reply(response: Any, result: type[T]) -> T:
    """Validate a chat completion into `result`; ValueError if it is unusable."""
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise LLMRefused("The model declined to answer (content_filter)")
    if choice.finish_reason == "length":
        raise ValueError("reply was cut off at max_tokens")
    text = choice.message.content
    if not text or not text.strip():
        raise ValueError("reply was empty; reply with the JSON object")
    return result.model_validate_json(text)


def chat_json(
    client: OpenAI,
    settings: Settings,
    messages: list[dict[str, Any]],
    result: type[T],
    max_tokens: int | None = None,
    model: str | None = None,
) -> T:
    """One structured agent call; retries once with the validation error, then LLMError.

    `max_tokens` (default `agent_max_tokens`) also bounds the model's reasoning tokens;
    `model` defaults to `agent_model`.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=model or settings.agent_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens or settings.agent_max_tokens,
        )
        try:
            return parse_reply(response, result)
        except (ValidationError, ValueError) as err:
            if attempt == MAX_ATTEMPTS:
                raise LLMError(
                    f"Invalid {result.__name__} after {attempt} attempts: {err}"
                ) from err
            logger.warning("Invalid %s (attempt %d), retrying: %s", result.__name__, attempt, err)
            messages = [
                *messages,
                {"role": "assistant", "content": response.choices[0].message.content or ""},
                {
                    "role": "user",
                    "content": f"That reply was invalid:\n{err}\nReply again with corrected JSON.",
                },
            ]
    raise AssertionError("unreachable")
