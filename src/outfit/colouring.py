"""Colour analysis of the profile photo: one DeepSeek vision call, only when the user asks.

DeepSeek gets a smaller re-encoded copy (no EXIF) and is asked only about colouring:
skin undertone and depth, hair, eyes, contrast, and the colours that follow from them.
Never ethnicity, age, gender, body shape or looks; body shape comes from what the user
types. The result is editable text (`profile.Colouring`); the photo is not used again.
"""

import base64
import logging

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from outfit.agents.llm import chat_json, load_prompt
from outfit.config import Settings
from outfit.profile import MAX_COLORS, Colouring, reencode

logger = logging.getLogger(__name__)

SEND_MAX_SIDE = 768  # enough to see colouring; smaller is cheaper and reveals less


class NoFaceVisible(ValueError):
    """The photo doesn't show a face clearly enough to judge colouring."""


class ColouringReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    face_visible: bool  # false: the rest is ignored
    summary: str = Field(default="", max_length=300)
    suits: list[str] = Field(default_factory=list, max_length=MAX_COLORS)
    go_easy: list[str] = Field(default_factory=list, max_length=MAX_COLORS)


def photo_message(photo: bytes) -> dict:
    jpeg = base64.standard_b64encode(reencode(photo, SEND_MAX_SIDE)).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg}"}},
            {"type": "text", "text": "Describe this person's colouring as instructed."},
        ],
    }


def analyse_colouring(photo: bytes, client: OpenAI, settings: Settings) -> Colouring:
    """One vision call; NoFaceVisible if the photo can't be judged, LLMError if no usable reply."""
    messages = [
        {"role": "system", "content": load_prompt("colouring", ColouringReply)},
        photo_message(photo),
    ]
    reply = chat_json(client, settings, messages, ColouringReply, model=settings.tagger_model)
    if not reply.face_visible or not reply.summary.strip():
        raise NoFaceVisible("No face is clearly visible; try a well-lit photo of your face")
    logger.info("Colouring analysed: %s", reply.summary)
    return Colouring(summary=reply.summary.strip(), suits=reply.suits, go_easy=reply.go_easy)
