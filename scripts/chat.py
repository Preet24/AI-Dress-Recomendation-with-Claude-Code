"""Chat with the outfit recommender in the terminal (one conversation thread).

Uses the real wardrobe, Open-Meteo and DeepSeek. Type 'quit' to stop, e.g.:
    python scripts/chat.py
    python scripts/chat.py --verbose      # also log what each agent does
    python scripts/chat.py --record       # save every turn to data/conversations/
"""

import argparse
import logging
import uuid

from outfit.agents.graph import build_graph, live_context
from outfit.agents.recording import Recorder, RecordingClient, run_turn
from outfit.config import Settings
from outfit.ingestion.tagger import make_client

logger = logging.getLogger("chat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--record", action="store_true", help="save each turn as JSON for prompt tuning"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger("outfit.agents").setLevel(logging.INFO)

    settings = Settings()
    client = RecordingClient(make_client(settings))
    graph = build_graph(client, settings, live_context(settings))
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    recorder = Recorder(settings.conversations_dir, thread_id, settings.agent_model)
    logger.info("What do you need an outfit for? (type 'quit' to stop)")

    paused = False  # waiting for the answer to a question
    while True:
        try:
            text = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        turn = run_turn(graph, config, text, resume=paused, client=client)
        paused = turn.question is not None
        logger.info("\nbot > %s", turn.question or turn.reply)
        if args.verbose:
            r = turn.record
            logger.info(
                "(%s · %d LLM calls · %d tokens · %.1fs)",
                " → ".join(r["path"]),
                r["llm_calls"],
                r["tokens"],
                r["seconds"],
            )
        if args.record:
            recorder.write(turn.record)

    if args.record and recorder.turns:
        logger.info("Recorded %d turn(s) to %s", recorder.turns, recorder.path)


if __name__ == "__main__":
    main()
