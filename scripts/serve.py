"""Serve the outfit recommender API (wardrobe + agent chat) on this machine.

One process only: local Qdrant lets a single process open the wardrobe, so stop
chat.py / review_wardrobe.py first. Interactive docs at http://127.0.0.1:8000/docs, e.g.:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --record   # also save chat turns to data/conversations/
"""

import argparse
import logging

import uvicorn

from outfit.api.app import create_app
from outfit.config import Settings

logger = logging.getLogger("serve")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--host", default="127.0.0.1", help="no login, so keep it local (default 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--record", action="store_true", help="save each chat turn as JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings()
    if args.record:
        settings = settings.model_copy(update={"record_chats": True})
    if args.host not in ("127.0.0.1", "localhost"):
        logger.warning("Serving on %s: anyone on that network can use the API", args.host)
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
