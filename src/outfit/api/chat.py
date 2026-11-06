"""Chat endpoints: talk to the agents; each turn ends with a reply or a question.

The client only sends messages. Whether the conversation is waiting for an answer
comes from the graph's own saved state (paused at the `ask` interrupt), so an answer
resumes the paused run, and a resume never repeats an LLM call.
"""

from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StringConstraints

from outfit.agents.recording import RecordingClient, run_turn
from outfit.api.errors import NotFound
from outfit.api.items import ItemOut
from outfit.api.services import Services, get_services
from outfit.tools.outfits import OutfitPlan

router = APIRouter(prefix="/chat", tags=["chat"])
ServicesDep = Annotated[Services, Depends(get_services)]
ThreadId = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class ChatIn(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    thread_id: ThreadId | None = Field(default=None, description="Omit to start a conversation")


class OutfitOut(BaseModel):
    pieces: list[ItemOut]
    reasons: list[str]
    warnings: list[str]


class ChatOut(BaseModel):
    thread_id: str
    kind: Literal["reply", "question"]
    text: str
    outfits: list[OutfitOut]  # the plan made this turn, best first; [] if none was made
    gaps: list[str]  # what the wardrobe is missing for that plan
    shown: list[ItemOut]  # items the user asked to see ("show me my tops")
    path: list[str]  # nodes this turn ran, e.g. ["router", "intake", "ask"]
    llm_calls: int
    seconds: float


class Conversation(BaseModel):
    thread_id: str
    messages: list[dict[str, str]]  # {"role": "user" | "assistant", "content": ...}
    # The bot's open question, if it is waiting for an answer. Not in `messages` yet:
    # the ask node adds it there only when the answer resumes the run.
    question: str | None


def outfits_of(plan: OutfitPlan | None) -> list[OutfitOut]:
    return [
        OutfitOut(
            pieces=[ItemOut.of(p) for p in outfit.pieces],
            reasons=list(outfit.reasons),
            warnings=list(outfit.warnings),
        )
        for outfit in (plan.outfits if plan else [])
    ]


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@router.post("")
def chat(body: ChatIn, s: ServicesDep) -> ChatOut:
    """Send a message, or the answer to the bot's question; returns this turn's result."""
    thread_id = body.thread_id or str(uuid4())
    config = config_for(thread_id)
    lock = s.thread_lock(thread_id)
    try:
        client = RecordingClient(s.client)
        graph = s.graph(client)
        state = graph.get_state(config)
        if body.thread_id and not state.values:
            raise NotFound(f"No conversation {thread_id} (the server may have restarted)")
        turn = run_turn(graph, config, body.message, resume=bool(state.next), client=client)
        if s.settings.record_chats:
            s.recorder(thread_id).write(turn.record)
    finally:
        lock.release()

    return ChatOut(
        thread_id=thread_id,
        kind="question" if turn.question else "reply",
        text=turn.question or turn.reply or "",
        outfits=outfits_of(turn.plan),
        gaps=turn.plan.gaps if turn.plan else [],
        shown=[ItemOut.of(item) for item in turn.shown],
        path=turn.record["path"],
        llm_calls=turn.record["llm_calls"],
        seconds=turn.record["seconds"],
    )


@router.get("/{thread_id}")
def conversation(thread_id: ThreadId, s: ServicesDep) -> Conversation:
    """The conversation so far, e.g. to redraw a chat after a page reload."""
    state = s.graph().get_state(config_for(thread_id))
    if not state.values:
        raise NotFound(f"No conversation {thread_id} (the server may have restarted)")
    question = str(state.interrupts[0].value) if state.interrupts else None
    return Conversation(
        thread_id=thread_id, messages=state.values.get("messages", []), question=question
    )
