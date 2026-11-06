"""Vision tagging with DeepSeek: one garment cutout -> the descriptive ClothingItem fields.

This is the only ingestion step that sends data off the machine, and it sends only
the garment cutout (flattened onto grey), never the original photo.

DeepSeek's API is OpenAI-compatible. Its JSON mode guarantees a JSON object but
not our schema, so the schema is spelled out in the system prompt and Pydantic
does the real validation. On invalid output the model gets the error and one
more try; after that, TaggingError.
"""

import base64
import io
import json
import logging
from functools import cache
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from outfit.config import Settings
from outfit.ingestion.embedder import flatten_on_background
from outfit.schemas import Category, Fit, Length, Occasion, Pattern, Scale

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "tagger.md"
SCHEMA_PLACEHOLDER = "{{schema}}"
MAX_ATTEMPTS = 2  # first try + one retry with the validation error


class TaggingError(RuntimeError):
    """The model did not return usable tags for a garment."""


class TagResult(BaseModel):
    """What the model decides about a garment; color LAB, id and paths are added later."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    subtype: str = Field(min_length=1)
    primary_color_name: str = Field(min_length=1)
    secondary_colors: list[str]
    pattern: Pattern
    fabric: str = Field(min_length=1)
    length: Length
    fit: Fit
    formality: int = Scale
    warmth: int = Scale
    flowy: bool
    occasions: list[Occasion] = Field(min_length=1)


def _enum(values: type) -> dict[str, Any]:
    return {"type": "string", "enum": [v.value for v in values]}


_SCALE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
_STRINGS = {"type": "array", "items": {"type": "string"}}

# Shown to the model in the system prompt (DeepSeek JSON mode does not enforce a
# schema); written out for readability, and a unit test keeps it in sync with TagResult.
TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": _enum(Category),
        "subtype": {"type": "string"},
        "primary_color_name": {"type": "string"},
        "secondary_colors": _STRINGS,
        "pattern": _enum(Pattern),
        "fabric": {"type": "string"},
        "length": _enum(Length),
        "fit": _enum(Fit),
        "formality": _SCALE,
        "warmth": _SCALE,
        "flowy": {"type": "boolean"},
        "occasions": {"type": "array", "items": _enum(Occasion)},
    },
    "required": list(TagResult.model_fields),
    "additionalProperties": False,
}


@cache
def load_prompt() -> str:
    """System prompt from prompts/tagger.md with the JSON schema filled in."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace(SCHEMA_PLACEHOLDER, json.dumps(TAG_SCHEMA, indent=2))


def make_client(settings: Settings) -> OpenAI:
    """DeepSeek client, using the key from `.env` / the environment.

    DeepSeek has no Python SDK of its own; its docs use the OpenAI SDK with DeepSeek's
    base_url, so requests go only to api.deepseek.com.
    """
    if settings.deepseek_api_key is None:
        raise TaggingError("DEEPSEEK_API_KEY is not set; add it to .env (see .env.example)")
    return OpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(), base_url=settings.tagger_base_url
    )


def encode_png(image: Image.Image) -> str:
    """Base64 PNG of the cutout on neutral grey (transparent pixels hide the old background)."""
    buffer = io.BytesIO()
    flatten_on_background(image).save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def first_messages(
    image: Image.Image, region: str, note: str | None = None
) -> list[dict[str, Any]]:
    text = f"Region hint: `{region}`. Tag this garment as JSON."
    if note:
        text += f"\nOwner's note: {note}"
    return [
        {"role": "system", "content": load_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_png(image)}"},
                },
            ],
        },
    ]


def parse_tags(response: Any) -> TagResult:
    """Validate a chat completion into TagResult; ValueError if it is unusable."""
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise TaggingError("The model declined to tag the garment (content_filter)")
    if choice.finish_reason == "length":
        raise ValueError("response was cut off at max_tokens")
    text = choice.message.content
    if not text or not text.strip():
        raise ValueError("response was empty; reply with the JSON object")
    return TagResult.model_validate_json(text)


def tag_garment(
    image: Image.Image,
    region: str,
    client: OpenAI,
    settings: Settings,
    note: str | None = None,
) -> TagResult:
    """Ask the model for the garment's tags; one retry on invalid output, then TaggingError.

    `note` is the owner's correction (e.g. "this is a midi slip dress"); it overrides
    what the image suggests.
    """
    messages = first_messages(image, region, note)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=settings.tagger_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=settings.tagger_max_tokens,
        )
        try:
            return parse_tags(response)
        except (ValidationError, ValueError) as err:
            if attempt == MAX_ATTEMPTS:
                raise TaggingError(f"Invalid tags after {attempt} attempts: {err}") from err
            logger.warning("Invalid tags (attempt %d), retrying: %s", attempt, err)
            messages = [
                *messages,
                {"role": "assistant", "content": response.choices[0].message.content or ""},
                {
                    "role": "user",
                    "content": f"That reply was invalid:\n{err}\nReply again with corrected JSON.",
                },
            ]
    raise AssertionError("unreachable")
