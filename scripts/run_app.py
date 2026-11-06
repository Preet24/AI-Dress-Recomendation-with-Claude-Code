"""Run the whole outfit recommender from one terminal: API + web UI, browser opens itself.

Ctrl+C stops both. Stop chat.py / review_wardrobe.py first (one process per wardrobe), e.g.:
    python scripts/run_app.py
    python scripts/run_app.py --port 8080 --ui-port 8502 --record --no-browser
"""

import argparse
import logging
import sys

from outfit.launcher import LaunchError, run

logger = logging.getLogger("run_app")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000, help="API port (default 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="UI port (default 8501)")
    parser.add_argument("--record", action="store_true", help="save each chat turn as JSON")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        code = run(args.port, args.ui_port, args.record, open_browser=not args.no_browser)
    except LaunchError as err:
        logger.error("%s", err)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
