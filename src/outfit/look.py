"""Look scores for a worn photo: one DeepSeek vision call, only when the user asks.

DeepSeek gets a smaller re-encoded copy of the wear-log photo (no EXIF), the pieces the
user said they wore, the occasion and, if known, their colouring summary. It scores the
outfit as worn (`wear_log.LookScores`: fit, proportion, colour, coordination, occasion,
polish, 1-5 with a reason, plus tips), never the person: the same rule as
`colouring.md`. No person in the photo -> `NoPersonVisible`, nothing saved.
"""

import base64
import logging

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, model_validator

from outfit.agents.llm import chat_json, load_prompt
from outfit.colouring import SEND_MAX_SIDE
from outfit.config import Settings
from outfit.profile import Colouring, reencode
from outfit.schemas import ClothingItem, Occasion
from outfit.tools.outfits import label
from outfit.wear_log import LookScore, LookScores

logger = logging.getLogger(__name__)

METRICS = ("fit", "proportion", "colour", "coordination", "occasion", "polish")


class NoPersonVisible(ValueError):
    """The photo doesn't show someone wearing the outfit clearly enough to score it."""


class LookReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_visible: bool  # false: the rest is left out
    fit: LookScore | None = None
    proportion: LookScore | None = None
    colour: LookScore | None = None
    coordination: LookScore | None = None
    occasion: LookScore | None = None
    polish: LookScore | None = None
    tips: list[str] = []

    @model_validator(mode="after")
    def _all_scored(self) -> "LookReply":
        if self.person_visible and (missing := [m for m in METRICS if getattr(self, m) is None]):
            raise ValueError(f"person_visible is true but these scores are missing: {missing}")
        return self


def outfit_text(
    pieces: list[ClothingItem], occasion: Occasion | None, colouring: Colouring | None
) -> str:
    """What the user said they wore, for the model to check the photo against."""
    lines = [
        "The user says they wore: " + ", ".join(label(p) for p in pieces) + ".",
        f"Occasion: {occasion.replace('_', ' ') if occasion else 'not given'}.",
        f"Their colouring: {colouring.summary}." if colouring else "Their colouring: unknown.",
        "Score this outfit as instructed.",
    ]
    return "\n".join(lines)


def photo_message(photo: bytes, text: str) -> dict:
    jpeg = base64.standard_b64encode(reencode(photo, SEND_MAX_SIDE)).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg}"}},
            {"type": "text", "text": text},
        ],
    }


def score_look(
    photo: bytes,
    pieces: list[ClothingItem],
    occasion: Occasion | None,
    colouring: Colouring | None,
    client: OpenAI,
    settings: Settings,
) -> LookScores:
    """One vision call; NoPersonVisible if nobody is shown, LLMError if no usable reply."""
    messages = [
        {"role": "system", "content": load_prompt("look", LookReply)},
        photo_message(photo, outfit_text(pieces, occasion, colouring)),
    ]
    reply = chat_json(
        client,
        settings,
        messages,
        LookReply,
        max_tokens=settings.tagger_max_tokens,  # six reasons, after the model's thinking
        model=settings.tagger_model,
    )
    if not reply.person_visible:
        raise NoPersonVisible("No one is clearly wearing the outfit in this photo")
    scores = LookScores(**reply.model_dump(include=set(METRICS)), tips=reply.tips[:2])
    logger.info("Look scored: %s", {m: getattr(scores, m).score for m in METRICS})
    return scores
