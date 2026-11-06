"""Eval scenarios: short scripted conversations with checkable expectations per turn.

Scenarios live in `evals/scenarios.toml` (stdlib `tomllib`, no new dependency) and are
validated here, so a typo in an expectation is an error instead of a silently skipped
check. A turn's `say` is a new message, or the answer when the previous turn asked a
question. Every scenario runs at the file's fixed `now` against one named weather
fixture, so the agents are the only thing that varies between runs.
"""

import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outfit.agents.intake import SAVED_TRIP
from outfit.agents.refine import TOOLS
from outfit.agents.state import Slots
from outfit.profile import Profile

SCENARIOS_PATH = Path(__file__).resolve().parents[3] / "evals" / "scenarios.toml"
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
Weather = Literal["warm_dry", "rainy", "cold"]


class Expect(BaseModel):
    """What must be true after a turn; unset fields are not checked."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["new", "follow_up", "chat"] | None = None  # the router's decision
    asks: bool | None = None  # True: the turn ends with a question; False: with a reply
    # The question asks for at least these (slot names, or "saved_trip": "same as usual?")
    missing: list[str] = Field(default_factory=list)
    # Request after the turn. Numbers match within 0.05, or give [min, max]; text
    # matches ignoring case, or give a list of accepted values; `start` matches any
    # spelling that parses to the same time ("tomorrow 9:00" = "tomorrow 09:00").
    slots: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)  # refine called at least these
    reply_mentions: list[str] = Field(default_factory=list)  # each; "a|b" = either
    reply_avoids: list[str] = Field(default_factory=list)
    plan_includes: list[str] = Field(default_factory=list)  # in every outfit's piece names
    plan_excludes: list[str] = Field(default_factory=list)  # in no outfit
    found_min: int | None = None  # items show_items put in front of the user
    names_top_outfit: bool = False  # the reply names every piece of option 1

    @field_validator("slots")
    @classmethod
    def _known_slots(cls, slots: dict[str, Any]) -> dict[str, Any]:
        unknown = set(slots) - set(Slots.model_fields)
        if unknown:
            raise ValueError(f"unknown slots: {sorted(unknown)}")
        return slots

    @field_validator("missing")
    @classmethod
    def _known_missing(cls, missing: list[str]) -> list[str]:
        unknown = set(missing) - set(Slots.model_fields) - {SAVED_TRIP}
        if unknown:
            raise ValueError(f"unknown slots: {sorted(unknown)}")
        return missing

    @field_validator("tools")
    @classmethod
    def _known_tools(cls, tools: list[str]) -> list[str]:
        unknown = set(tools) - TOOL_NAMES
        if unknown:
            raise ValueError(f"unknown tools: {sorted(unknown)}")
        return tools


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str = Field(min_length=1)
    expect: Expect = Field(default_factory=Expect)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    about: str
    tags: list[str] = Field(default_factory=list)
    weather: Weather = "warm_dry"
    profile: Profile = Field(default_factory=Profile)  # the user's profile; default empty
    turns: list[Turn] = Field(min_length=1)


class ScenarioSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    now: datetime  # the user's clock for every scenario
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ScenarioSet":
        ids = [s.id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return self


def load_scenarios(path: Path = SCENARIOS_PATH) -> ScenarioSet:
    with path.open("rb") as f:
        return ScenarioSet.model_validate(tomllib.load(f))
