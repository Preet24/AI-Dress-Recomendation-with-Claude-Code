# Phase 5 — Agent evaluation

Phase 4 built the agents and checked them by hand in a few live chats. Phase 5 makes
that a measurement: 21 scripted conversations, run against the real DeepSeek model
with everything else fixed, checked automatically, and summarized per agent. Then
fix what the numbers show, and re-measure.

The point is not a high score; it is knowing *why* a run fails. Every fix below came
from reading a recorded turn, and only three of the seven were prompt wording; the
rest were tests, code and config.

## What it delivers

| Piece | File | Job |
|---|---|---|
| Recorder | `agents/recording.py`, `scripts/chat.py --record` | Capture every node's update and every LLM call in a turn |
| Scenarios | `evals/scenarios.toml`, `evals/scenarios.py` | 36 conversations with checkable expectations per turn (10 added later from a real failed chat and more occasions, places and times: dates, parent-teacher meeting, long drive, sightseeing; tag `expanded`; 5 with a user profile: saved places and the home city, tag `profile`) |
| Checker | `evals/checks.py` | One pass/fail result per expectation, in plain words |
| Fixed inputs | `evals/fixtures.py`, `evals/wardrobe.json`, `scripts/snapshot_wardrobe.py` | Wardrobe snapshot, weather fixtures, a fixed clock |
| Runner | `evals/runner.py`, `scripts/eval_agents.py` | Run scenarios in parallel, report per agent, flag flaky ones |

## 1. Recording a turn

The graph's last node, `respond`, resets the per-turn fields (the draft, the critic's
issues, the tool calls) so the next message starts clean. The final state therefore
cannot tell you what happened. `run_turn` streams the graph with
`stream_mode="updates"`, capturing each node's update as it happens, and
`RecordingClient` wraps the DeepSeek client to log each call: input (without the
system prompt), output, finish reason, tokens, time. Nodes run one at a time, so a
call belongs to the node whose update follows it. No graph code changed.

Each turn is tagged with a hash of all prompt files, so recordings and eval results
from before and after a prompt change can be told apart.

## 2. Scenarios

Each scenario is a short conversation; a turn says something and lists what must be
true afterwards. Unset expectations are not checked, and the schema rejects unknown
keys, slots and tools, so a typo is an error rather than a check that silently never
runs.

```toml
[[scenarios]]
id = "swap_to_jeans"
about = "A requested swap: find the jeans, re-plan with them, pass on the warning."
tags = ["refine"]

[[scenarios.turns]]
say = "Party in Pune tonight at 7, about 4 hours, half an hour outside, going by cab"

[[scenarios.turns]]
say = "something with my blue jeans instead"
[scenarios.turns.expect]
intent = "follow_up"
tools = ["update_request"]
plan_includes = ["jeans"]
reply_mentions = ["jeans", "casual|warm|warmer|dressy|formal|relaxed"]
```

Expectations check **outcomes**, never exact text: the router's intent, whether the
turn ends in a question and what it asks for, request slots (times compared by
meaning, so "tomorrow 9:00" = "tomorrow 09:00"; numbers within 0.05 or a range),
tools called, what every or no outfit contains, words in the reply (`a|b` = either),
items shown, and whether the reply names every piece of option 1.

| Group | Scenarios |
|---|---|
| Intake (11) | all at once · vague then answer · partial office request by scooter · "in 2 hours" · cycling = bicycle · "my bike" = motorbike · skip weather · interview = formality 4–5 · item named in the request · item ruled out · give up after 3 questions |
| Router (4) | "same city" back-reference · a wedding replaces the party · "thanks" is chat · a greeting first |
| Refine (4) | swap to jeans · show my tops · a saree I don't own · something more relaxed |
| Weather (2) | rain on a scooter · cold evening, no outerwear |

## 3. Fixed inputs

So that only the agents vary between runs:

- **Wardrobe**: `scripts/snapshot_wardrobe.py` copies the real store (10 items with
  their CLIP vectors) to `evals/wardrobe.json`; each scenario restores its own
  in-memory Qdrant. CLIP text search stays real.
- **Weather**: three fake Open-Meteo fixtures (`warm_dry`, `rainy`, `cold`), each set
  clearly inside one band of the advice rules so a small change never flips them.
- **Clock**: `Context.now` (new, `None` = real time) fixes "now" to Saturday 16:00 for
  intake, the refine tools and the weather window, so "tonight" and "Monday at 9"
  always mean the same thing.

## 4. The runner and the report

`scripts/eval_agents.py` gives each scenario a fresh graph, thread and store (so 4
run in parallel), answers a question when the previous turn asked one, checks every
turn and prints:

```
Scenarios passed: 59/63 (94%)
Checks by agent:
  router       intent            39/39 (100%)
  intake       slots             149/150 (99%)
  ...
Turns: 99 (77 styled replies)
  critic revisions   5/77 (6%)
  template fallbacks 0/77 (0%)
  JSON retries 2 · truncated replies 2 · API errors 0 · turn errors 0
  LLM calls 358 · tokens 579226 · 6.1s per turn (max 18.58s)
Flaky (passed some runs): skip_weather 2/3, give_up_after_three_questions 1/3, swap_to_jeans 2/3
```

It always prints a cost estimate and asks before spending API calls (`--yes` to skip;
no terminal means no). `--only` / `--tag` select scenarios, `--repeat N` exposes
flakiness. Full results, with every recorded turn, go to `data/evals/` (gitignored).

## 5. Results

### Before any agent run: a code bug found for free

Running the deterministic part (orchestrate on the snapshot, real CLIP, no LLM) for
the scenarios' expected requests showed that **item references resolved by CLIP
alone are unreliable** on real photos:

| Reference | CLIP's best match | Problem |
|---|---|---|
| "lavender skirt" | beige trousers 0.243 (skirt 0.226) | wrong item accepted |
| "black sequin top" | black sequin top 0.231 | right item, under the 0.24 threshold: "not found" |

Fix: `tools/match.py` matches the user's words against the item's tags first (the
reference's head noun must be the item's garment word; most words matched wins;
CLIP only breaks ties or handles references no tag matches). Used by both orchestrate
and refine's `search_wardrobe`; 12 real-wardrobe references are now tests.

### The runs

| Run | Prompts | Scope | Passed | Critic revisions | DeepSeek calls |
|---|---|---|---|---|---|
| 1. Baseline | `668db50c` | all 21 × 1 | 20/21 | 1/26 | 118 |
| 2. Stability | `668db50c` | all 21 × 3 | 59/63 | 5/77 | 358 |
| 3. After fixes 2–5 | `12f68250` | 3 flaky × 3 | 9/9 | 3/12 | 66 |
| 4. After fix 6 | `cd9fd217` | `swap_to_jeans` × 3 | 3/3 | 0/6 | 30 |
| 5. Full re-run | `cd9fd217` | all 21 × 3 | **63/63** | 11/78 | 368 |
| 6. After fix 7 (final) | `3a5c446a` | all 21 × 3 | **63/63** | **1/78** | 354 |

Runs 3 and 4 re-ran only the affected scenarios; run 5 re-ran everything on the
prompts after fixes 1–6. Every check passed (router 39/39, intake 225/225,
orchestrate 18/18, refine 15/15, stylist 48/48), nothing was flaky, and there were
no template fallbacks or API errors. But critic revisions doubled (5 → 11) and the
critic thought much longer (reasoning tokens p90 854 → 2266; one call used all 4096
twice, so its rubric was skipped and the draft was sent). That led to fix 7.

Run 6, on the final prompts, keeps 63/63 with nothing flaky and brings revisions down
to 1/78; that one is a real catch (the draft called two clashing patterns "playing
off each other nicely" when `why` says they compete). Against the baseline (run 2):

| | Run 2 (baseline) | Run 6 (final) |
|---|---|---|
| Scenarios passed | 59/63 | 63/63 |
| Checks failed | 6 | 0 |
| Critic revisions | 5/77 | 1/78 |
| JSON retries · truncated replies | 2 · 2 | 0 · 0 |
| Critic reasoning tokens, p90 · max | 854 · 2048 (cut off) | 641 · 2178 |
| LLM calls · tokens | 358 · 579k | 354 · 601k |
| Seconds per turn, mean · max | 6.1 · 18.6 | 6.5 · 17.1 |

The stylist now reasons a little longer (median 176 → 265 tokens, max 1362 of its
2048 budget), the price of the extra rules; worth watching if prompts grow.

Per agent in the stability run (2): router 39/39, intake 223/225 (asks 65/66, missing
9/9, slots 149/150), orchestrate 18/18, refine 15/15, stylist 44/48. Template
fallbacks 0, API errors 0, 6.1 s per turn on average.

### What failed, and the fixes

| # | Finding (from the recordings) | Kind | Fix |
|---|---|---|---|
| 1 | `exclude_an_item`: "reply mentions 'jeans'", but only to confirm "not the jeans" meant the medium blue jeans, which it should | test | removed the wrong `reply_avoids`; `plan_excludes` already checks the outfits |
| 2 | `give_up_after_three_questions` (1/3): intake itself set `skip_weather` on "no idea", so the give-up step had no weather note to add, and the reply never said weather was ignored | code | planning without weather always adds a weather assumption (`orchestrate.NO_WEATHER`), once |
| 3 | `skip_weather` (2/3): "I don't know the venue or the timing yet" → intake asked about the trip instead of skipping weather | prompt | `intake.md`: that case named in the rule and given as an example |
| 4 | `swap_to_jeans` (2/3): the reply passed the warning on ("aren't tagged for party wear and run warmer") without the word "casual" | test | expectation accepts the ways the warning can be put |
| 5 | 2 critic calls returned nothing: `reasoning_tokens` 2048 = the whole budget spent thinking (the retry recovered) | config | `critic_max_tokens = 4096` for the rubric only (see below) |
| 6 | Run 3: all 3 critic revisions objected to "the third pick" / "the second option"; both prompts had the rule, worded vaguely ("ranking numbers") | prompt | `stylist.md` and `critic.md` spell it out: no option numbers or ranking words, introduce alternatives by their pieces; "another option" is fine |
| 7 | Run 5: all 11 revisions were real catches, but 9 came from the two prompts disagreeing. The stylist's own examples invited "for something dressier, …" and "a light jacket would help", while the critic counts a reason not in `why` and an extra piece not in `gaps` as errors (3 extra reasons, 6 extra-piece tips such as "flatter shoes", "sun protection"). The skipped rubric was the critic spending 4096 tokens deciding whether "dressier" is a reason, and it missed an invented "Assuming you're leaving now". Twice "half an hour outside" became "each way" | prompt | `stylist.md`: "a …" only for a `gaps` item, no reasons beyond `why`, only the given assumptions, outdoor time is the total; alternatives introduced as "if you'd rather wear a dress". `critic.md`: the same rules, so both prompts agree |

Fix 5 was sized from the recorded token usage, not guessed. DeepSeek counts reasoning
tokens against `max_tokens`, and only the critic has a long tail:

| Agent | Calls | Reasoning tokens, median | p90 | max |
|---|---|---|---|---|
| router | 84 | 12 | 42 | 98 |
| intake | 81 | 41 | 132 | 296 |
| refine | 32 | 25 | 147 | 200 |
| stylist | 82 | 176 | 396 | 761 |
| critic | 79 | 321 | 854 | **2048** (cut off) |

### What the numbers say

- **Routing and extraction are solid**: 100% intent, 99% slots over 3 runs, including
  relative times, "same city", Indian English "bike", and give-up after 3 questions.
- **The reflection loop earns its place**: every revision read across the runs was a
  real catch (a wrong "each way", an invented tip, a reversed reason), and the
  template fallback was never needed.
- **Most failures were not the model's**: of seven fixes, two were tests, one was
  code, one was config, and three were prompt wording. Reading recorded turns before
  touching prompts is what made that visible.
- **Pass rate alone would have hidden fix 7**: run 5 scored 63/63, while the
  revision rate and reasoning tokens showed the stylist and critic prompts
  contradicting each other. Two prompts that judge each other must state the same rules.

## Next

- Harder scenarios: the current set is near 100%, so it no longer separates prompt
  versions (ambiguous follow-ups like "and if it rains?", several changes in one
  message, contradictions).
- When nothing fits, say *which* rule removed the candidates ("both office bottoms
  are flowy, which riding rules out"): the stylist currently gets only the generic
  gap and once guessed "the heat".

## Commands

```bash
python scripts/chat.py --record --verbose            # record live turns to data/conversations/
python scripts/snapshot_wardrobe.py                  # refresh evals/wardrobe.json after ingesting
python scripts/eval_agents.py                        # all scenarios (asks first)
python scripts/eval_agents.py --tag refine --repeat 3
python scripts/eval_agents.py --only swap_to_jeans,show_my_tops --yes
```

## Testing

The eval machinery itself is unit-tested with the fake client (no API): recording
through an interrupt and resume, scenario validation and typo rejection, every check
kind passing and failing, the fixtures (weather bands, snapshot round trip, the fixed
clock reaching intake and the weather), the runner's metrics (revision, template
fallback, JSON retry, a crashing turn), and the flaky-run report. Files:
`test_recording.py`, `test_eval_scenarios.py`, `test_eval_fixtures.py`,
`test_eval_runner.py`, `test_match.py`.
