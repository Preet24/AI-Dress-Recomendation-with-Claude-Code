"""Run the agent eval scenarios against the real DeepSeek API and report per-agent scores.

Uses the fixed inputs (evals/wardrobe.json, weather fixtures, the scenario file's
clock), so only the agents vary. Spends real API calls: shows an estimate and asks
first unless --yes. Full results go to data/evals/ (gitignored), e.g.:
    python scripts/eval_agents.py                      # all scenarios
    python scripts/eval_agents.py --tag refine         # one group
    python scripts/eval_agents.py --only swap_to_jeans,show_my_tops --yes
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from outfit.agents.recording import jsonable, prompts_version
from outfit.config import Settings
from outfit.evals.fixtures import eval_context, load_snapshot, restore
from outfit.evals.runner import format_report, run_scenario, summarize
from outfit.evals.scenarios import load_scenarios
from outfit.ingestion.embedder import get_clip
from outfit.ingestion.tagger import make_client

logger = logging.getLogger("eval")

CALLS_PER_TURN = 4  # router/intake + stylist + critic, more for refine: a rough estimate
RESULTS_DIR = Path("data/evals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--tag", help="only scenarios with this tag")
    parser.add_argument("--repeat", type=int, default=1, help="run each scenario N times")
    parser.add_argument("--workers", type=int, default=4, help="scenarios run in parallel")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending API calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    scenario_set = load_scenarios()
    scenarios = [
        s
        for s in scenario_set.scenarios
        if (not args.only or s.id in args.only.split(",")) and (not args.tag or args.tag in s.tags)
    ] * args.repeat
    if not scenarios:
        raise SystemExit("No scenarios match.")
    turns = sum(len(s.turns) for s in scenarios)
    logger.info(
        "%d scenario run(s), %d turns: roughly %d DeepSeek calls.",
        len(scenarios),
        turns,
        turns * CALLS_PER_TURN,
    )
    if not args.yes and not confirm("Run them? [y/N] "):
        raise SystemExit("Cancelled.")

    settings = Settings()
    client = make_client(settings)
    snapshot = load_snapshot()  # read once; each scenario gets its own in-memory store
    get_clip(settings.clip_model, settings.clip_pretrained)  # load once, before the threads

    def play(scenario):
        store = restore(snapshot)
        result = run_scenario(
            scenario,
            client,
            settings,
            lambda weather: eval_context(store, settings, weather, scenario_set.now),
            scenario_set.now,
        )
        logger.info("%s %s", "PASS" if result.passed else "FAIL", scenario.id)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(play, scenarios))

    summary = summarize(results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}_{prompts_version()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "prompts": prompts_version(),
        "model": settings.agent_model,
        "summary": summary,
        "scenarios": [
            {
                "id": r.scenario.id,
                "passed": r.passed,
                "turns": [
                    {
                        "say": t.say,
                        "failures": t.failures,
                        "metrics": t.metrics,
                        "record": t.record,
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(jsonable(full), ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("\n%s\n\nFull results: %s", format_report(summary), out)


def confirm(question: str) -> bool:
    try:
        return input(question).strip().lower() == "y"
    except EOFError:  # no terminal to ask: don't spend
        return False


if __name__ == "__main__":
    main()
