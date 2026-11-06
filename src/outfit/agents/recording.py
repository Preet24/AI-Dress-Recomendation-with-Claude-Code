"""Record live conversations for prompt tuning: what every node did, turn by turn.

`run_turn` runs one graph turn with `stream_mode="updates"`, so each node's state
update is captured as it happens (respond resets the per-turn fields, so the final
state alone would lose the draft, the critic's issues and the tool calls).
`RecordingClient` wraps the DeepSeek client and logs each call's input, output,
tokens and time; nodes run one at a time, so a call belongs to the node whose
update follows it. `Recorder` appends one JSON line per turn to
`<conversations_dir>/<start>_<thread>.jsonl`.
"""

import dataclasses
import hashlib
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel

from outfit.agents.llm import PROMPTS_DIR
from outfit.schemas import ClothingItem
from outfit.tools.outfits import OutfitPlan, label


def jsonable(value: Any) -> Any:
    """Plain JSON data for anything in graph state or an SDK reply.

    Garments become a short reference (id prefix, name, category): a recording is read
    by a person tuning prompts, and the full tags are in the wardrobe.
    """
    if isinstance(value, ClothingItem):
        return {"id": str(value.id)[:8], "name": label(value), "category": value.category.value}
    if isinstance(value, BaseModel):  # our schemas and the openai SDK's reply objects
        return value.model_dump(mode="json", exclude_none=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, SimpleNamespace):
        return jsonable(vars(value))
    return str(value)  # UUID, datetime, Path


def prompts_version() -> str:
    """Short hash of all prompt files, to tell recordings from before and after a change."""
    digest = hashlib.sha256()
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()[:8]


@dataclasses.dataclass
class LLMCall:
    seconds: float
    tools: bool  # a tool-calling (refine) request rather than JSON mode
    request: list  # the messages after the system prompt
    finish_reason: str | None = None
    usage: dict | None = None
    response: Any = None
    error: str | None = None


class RecordingClient:
    """The DeepSeek client, logging every chat completion it makes."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.calls: list[LLMCall] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        request = [m for m in kwargs.get("messages", []) if m.get("role") != "system"]
        call = LLMCall(seconds=0.0, tools="tools" in kwargs, request=jsonable(request))
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as err:
            call.error = f"{type(err).__name__}: {err}"
            raise
        finally:
            call.seconds = round(time.perf_counter() - start, 2)
            self.calls.append(call)
        choice = response.choices[0]
        call.finish_reason = choice.finish_reason
        call.usage = jsonable(getattr(response, "usage", None))
        call.response = jsonable(choice.message)
        return response

    def take(self) -> list[LLMCall]:
        """The calls made since the last `take`."""
        calls, self.calls = self.calls, []
        return calls


@dataclasses.dataclass
class Turn:
    """One graph run: it ends with a reply, or pauses with a question."""

    reply: str | None
    question: str | None
    record: dict
    plan: OutfitPlan | None = None  # the plan made this turn (None if it made none)
    shown: list[ClothingItem] = dataclasses.field(default_factory=list)  # refine's show_items


def run_turn(
    graph: CompiledStateGraph,
    config: dict,
    text: str,
    resume: bool,
    client: RecordingClient | None = None,
) -> Turn:
    """Send a user message (or the answer to a question) and capture every node's update."""
    payload = Command(resume=text) if resume else {"messages": [{"role": "user", "content": text}]}
    steps: list[dict] = []
    question = None
    plan, shown = None, []
    start = last = time.perf_counter()
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        now = time.perf_counter()
        for node, update in chunk.items():
            if node == "__interrupt__":
                question = str(update[0].value)
                continue
            # Kept raw: respond resets `found`, and the router clears `plan` on a new request.
            plan = (update or {}).get("plan") or plan
            shown = (update or {}).get("found") or shown
            calls = client.take() if client else []
            steps.append(
                {
                    "node": node,
                    "seconds": round(now - last, 2),
                    "update": jsonable(update or {}),
                    "llm": [jsonable(c) for c in calls],
                }
            )
        last = now

    reply = None if question else graph.get_state(config).values.get("reply")
    usages = [c["usage"] for s in steps for c in s["llm"] if c.get("usage")]
    record = {
        "user": text,
        "resumed": resume,
        "seconds": round(time.perf_counter() - start, 2),
        "path": [s["node"] for s in steps],
        "llm_calls": sum(len(s["llm"]) for s in steps),
        "tokens": sum(u.get("total_tokens", 0) for u in usages),
        "question": question,
        "reply": reply,
        "steps": steps,
    }
    return Turn(reply, question, record, plan, shown)


class Recorder:
    """Appends turns of one conversation to a JSON-lines file."""

    def __init__(self, directory: Path, thread_id: str, model: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"{stamp}_{thread_id[:8]}.jsonl"
        self.header = {"thread_id": thread_id, "model": model, "prompts": prompts_version()}
        self.turns = 0

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.turns += 1
        line = {"turn": self.turns, "at": datetime.now().isoformat(timespec="seconds")}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**line, **self.header, **record}, ensure_ascii=False) + "\n")
