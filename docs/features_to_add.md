# Features to add

Planned features, agreed in discussion before any code is written. Update this file
as decisions are made and steps are built (status per step: planned / in progress /
done).

Order: **1. User profile** first, then **2. Shopping assistant**, which builds on it,
then **3. Accessories**, then **4. Trip packing list** and **5. Wear feedback**
(requested 2026-10-25).

---

## 1. User profile

**Status:** done (steps 1–5; step 5's live check needs a photo from the user)

The user tells the app about themselves and the places they go. Rule: every field
must change something the app does; no field is stored "just in case".

### About you (every field optional)

| Field | Example | What it changes |
|---|---|---|
| Gender / style direction | man, woman, non-binary, rather not say | reply wording; what the shopping assistant suggests (no heels unless they fit the style) |
| Home city | Bangalore | default city when none is said ("office tomorrow"); the shopping assistant's typical climate |
| Height, build / body shape | 178 cm, athletic / slim / broad / … | fit tips in replies ("relaxed tops balance broad shoulders"); shopping advice |
| Waist / sizes | waist 32, shirt M | shopping only ("look for chinos in 32") |
| Fit preferences | slim trousers, relaxed shirts | outfit ranking: preferred fits score a little higher |
| Colors liked / avoided | likes navy, olive; avoids yellow | ranking: avoided colors score lower, never banned |

### Your places (an editable list)

| Place | Dress code | Usual trip |
|---|---|---|
| Office | business casual (formality 3–4) | 9:00, 9 h, 15 min outside, by cab |
| Gym | sports | 7:00, 1.5 h, by scooter |
| College | casual | … |

Dress code choices map to formality: formal (4–5), business (4), business casual
(3–4), smart casual (3), casual (2–3).

"Office tomorrow" then takes the dress code, start time, hours and transport from
the saved place and uses the user's real dress code instead of the generic office
range. It **confirms first** ("same as usual: 9:00, 9 h, 15 min outside, cab?"): one
extra turn, but it catches unusual days. A "yes" plans with the saved trip; any
other answer is read as changes to it.

### Photo (the sensitive part)

Two possible uses:

1. **Display only**: shown on the profile page, kept local. Harmless, adds little.
2. **Color analysis**: DeepSeek describes skin tone, hair and eye color → a personal
   palette ("warm tones such as olive, rust and cream suit you"), used in outfit
   ranking and the shopping list. Requires sending a face photo to DeepSeek.

Decided: opt-in color analysis.

- the photo is optional and stored only locally (gitignored);
- it is sent to DeepSeek only when the user clicks **Analyse my colouring**, with a
  clear note that it will be sent;
- the result is editable text ("warm, medium skin tone"); only that text is used
  afterwards, never the photo again;
- body shape always comes from what the user types, never from the photo.

### How it's built

- **Storage:** one profile in `data/profile.json` (gitignored, `Settings.profile_path`),
  the Pydantic `Profile` in `src/outfit/profile.py`, saved through `ProfileStore`
  (temp file swapped in, so never half-written); one user per install, no login.
- **API:** `GET/PUT /profile` (`src/outfit/api/profile.py`; PUT replaces the whole
  profile). Step 5 adds photo upload and `POST /profile/analyse`.
- **Schema:** gender (man / woman / non_binary), home city, height (cm), build (slim,
  average, athletic, broad, curvy, plus_size), waist, top size, fit for tops and for
  bottoms, colors liked / avoided (trimmed, lower-case), and up to 20 saved places
  (unique names; occasion, optional dress code, usual start, hours away / outdoors,
  transport).
- **UI:** a third page, **Profile**, next to Chat and Wardrobe.
- **Agents:** intake fills empty slots from the profile and from a named saved place;
  outfit rules use dress codes, fit and color preferences; the stylist gets only the
  profile facts relevant to the reply.
- **What leaves the machine:** only the profile facts a reply needs (in the stylist's
  facts), plus the photo if the user clicks Analyse. The user guide must say so.

### Build steps

| # | Step | Status |
|---|---|---|
| 1 | Profile schema + storage + API | done |
| 2 | Profile page in the UI (`ui/profile_page.py`) | done |
| 3 | Saved places and home city in chat (intake) | done (live eval 2026-10-22, tag `profile`: 5/5, 25/25 slot checks) |
| 4 | Fit and color preferences in outfit ranking (`taste()` in `tools/outfits.py`) | done |
| 5 | Photo color analysis (opt-in; `outfit/colouring.py`, `prompts/colouring.md`, `/profile/photo`, `/profile/analyse`) | done (live check pending) |

### Decisions (2026-10-22)

1. Photo: opt-in color analysis, as proposed.
2. Saved places: confirm first, not silently.
3. Age: dropped (it changed too little).
4. Build order: as above.

---

## 2. Shopping assistant

**Status:** done (Part 2, then Part 1; live check of Part 2 passed 2026-10-23)

Two parts that share most of their code.

### Part 1: "What should I buy?" (wardrobe gaps)

Runs the existing planner (`suggest_outfits`) for every occasion across a few typical
weathers (hot, mild, cold, rainy), collects which requests fail and why (the
`closest_misses` gaps already say which rule removed which garment), and ranks
purchases by how much they would unlock.

Example for the real wardrobe:

| Buy | Why | Unlocks |
|---|---|---|
| A pair of shoes (e.g. white sneakers, brown loafers) | no footwear at all: every outfit is incomplete | every occasion |
| Dressy trousers (formality 4) | nothing for formal dates, weddings, formal office | 3 occasions |
| A light jacket or layer | cold or rainy outings have no outer layer | cold, rain |
| A light linen shirt | most tops are warmth 2+ cotton; better for 35 °C | hot days |

Ranking and reasons are deterministic Python, so every suggestion is backed by a
failed outfit or a missing piece. DeepSeek only writes the friendly summary
(optional).

### Part 1 build steps (requested 2026-10-23: "what should I buy for a selected occasion, and what is missing")

How: a catalogue of ~30 staples (`tools/staples.toml`: tags + usual colors, filtered by
the profile's gender) each go through Part 2's `check_candidate` in the color that goes
with most of the wardrobe (avoided / go-easy / already-owned colors skipped); picked
greedily (each pick is added before scoring the rest), shoes and outerwear only for an
unlock or a gap, a top + bottom pair when only both together unlock a situation.
Local and free: no DeepSeek call.

| # | Step | Status |
|---|---|---|
| A | Scoring (`tools/staples.py`: `missing`, `suggest_purchases`; `shopping.baseline`, grouped "in any weather" unlocks; `ryb_hue` cached, 9 s -> 1.3 s on the snapshot) | done |
| B | API `POST /shopping/suggest {occasions}`, "What should I buy?" tab on the Shopping page, docs | done |

### Part 2: "Should I buy this?" (check a candidate)

The user uploads a photo of something they might buy (shop photo, screenshot, photo
in the store). The app:

1. **Identifies it**: cutout + tagging, the same pipeline as the wardrobe, but **not
   saved**.
2. **Pairs it**: which of the user's own pieces it goes with (color harmony + outfit
   rules), and what clashes.
3. **Counts new outfits**: outfits with vs without it ("adds 14 outfits, and makes
   office wear in the heat possible"), the most useful number.
4. **Checks for duplicates**: CLIP "looks alike" ("very close to your white button-up
   shirt").
5. **Checks gaps**: does it fix one from Part 1?
6. **Verdict**: buy / maybe / skip, with reasons.

Bought it? **Add to wardrobe** stores it like an upload, without tagging it again.

### Where it lives

- A new **Shopping** page: "What to buy" and "Check an item".
- Two API endpoints (gap list, candidate check).
- Later, maybe: a chat tool ("what should I buy for weddings?", "would this jacket
  work?").

### Cost

- What to buy: local and free (an optional summary = 1 DeepSeek call).
- Check an item: 1 tagging call per photo + ~30 s CPU segmentation.

### Decisions (2026-10-23)

1. Occasions: the saved places' occasions plus occasions the user ticks on the
   Shopping page (stored in the profile); casual, office, party if neither is set.
2. Climate: the home city over the whole year (last year's daily weather from the
   free Open-Meteo archive decides which of hot / mild / cold / rainy occur); all four
   without a home city.
3. Photo mode: the user chooses, preset to "worn".
4. Suggestions say why a color ("white goes with 15 of your 20 items").
5. Budget and brands: out of scope.
6. Order: Part 2 first, then Part 1 reusing its scoring.

### Part 2 build steps

| # | Step | Status |
|---|---|---|
| 1 | Pipeline split: describe a photo without storing (`describe_photo`); candidates held in memory (`ingestion/candidates.py`), "add" stores them | done |
| 2 | Scoring (`tools/shopping.py`, `tools/climate.py`, `all_outfits()`, `Profile.occasions`): pairs, new outfits, unlocks, duplicates, verdict | done |
| 3 | API (`api/shopping.py`): `POST /shopping/check`, candidate image, `POST /shopping/candidates/{id}/add` | done |
| 4 | Shopping page (`ui/shopping_page.py`): occasion ticks, "Check an item" report cards, Add to wardrobe | done |
| 5 | Docs (`docs/phases/shopping-assistant.md`), duplicate threshold calibrated (0.85 / similar 0.75), live check | done except the live check (needs a shop photo from the user) |
---

## 3. Accessories

**Status:** done, live-checked 2026-10-24 (requested 2026-10-23: "glasses, ties, belt, purse etc"; explainer: `docs/phases/accessories.md`)

The `accessory` category is already tagged (bags, belts, scarves, jewellery, hats), but
outfits ignore it. Accessories are added to an outfit **after** it is chosen, at most
one per kind, so they never change which outfits are suggested.

### Kinds and rules

The kind comes from the subtype words (`tools/accessories.py`), not a new
`ClothingItem` field, so stored items and the tagger stay as they are.

| Kind | Added when |
|---|---|
| Belt | the bottom is trousers, jeans or chinos (not track pants, leggings or ethnic bottoms); matches the shoes' leather (black with black, brown with brown) |
| Tie | the occasion allows formality 4+ (office, party, festive, formal, wedding) and the top is a shirt (not a t-shirt, polo or kurta); color goes with the shirt, no two busy patterns |
| Bag | its formality and occasions fit (a clutch for a party, a tote for the office, a backpack for travel); rain-sensitive fabrics lose fit as for any piece |
| Sunglasses | only in strong sun with time outdoors, or when riding |
| Hat | only in strong sun or cold |
| Scarf / stole | only in the cold |
| Dupatta | only with an ethnic top or bottom |
| Watch, jewellery | formality and occasions fit |

Shared: flowy accessories are out when the weather or ride rules out flowy pieces (a
dupatta on a scooter); a hat or scarf more than one point warmer than the weather is
out; score = half color harmony with the outfit, half fit, plus taste; added at ≥ 0.6.
Missing ones are a gap only when called for: sunglasses in strong sun, a scarf in
the cold, and a tie for formal or wedding when the user owns ties but none
fits.

Accessories need a photo of the item alone (worn photos give only top and bottom).
Prescription glasses are not an outfit choice: later, maybe a profile setting
(frame color in color matching).

### Build steps

| # | Step | Status |
|---|---|---|
| 1 | Kinds and rules (`tools/accessories.py`: `kind_of`, `choose_accessories`; `WeatherAdvice.sunny` / `riding`) | done |
| 2 | Accessories in outfits (`finish_plan` after `suggest_outfits`, in orchestrate and `scripts/suggest_outfits.py`; score unchanged) and their gaps | done |
| 3 | Stylist and critic prompts (accessories come last in `pieces`, named after the clothes; listed ones are not additions), critic word lists (accessory nouns, accessory gaps); API and UI cards show them as pieces with no change | done (prompts changed: live eval not re-run) |
| 4 | Shopping: accessory candidates (`accessory_effect`: outfits it joins, accessory gaps it closes; buy only with a fix), `accessory_gaps` in the missing list, staples sunglasses / wool scarf / silk tie (man), clothes before accessories at equal fixes; typical days get their own UV (only hot and mild are "strong sun") | done |
| 5 | Docs (`docs/phases/accessories.md`, user guide, README, phase 3/4 docs, shopping-assistant, agentic-architecture), CLAUDE.md; live check with the user's accessory photos (asks first) | done (live check 2026-10-24: belts and sunglasses as intended; fixed socks tagged as bottom/footwear and sunglasses judged by occasion; ties untested, no formal outfit yet) |

---

## 4. Trip packing list

**Status:** done, live-checked 2026-10-25 (requested 2026-10-25; explainer: `docs/phases/packing.md`)

The user describes a trip ("Goa, 4 days from Friday, two dinners out, one beach
day") and gets the **fewest pieces that cover every day**, plus which outfit to wear
on which day. Everything is deterministic Python on top of the existing planner; no
LLM decides what goes in the bag.

### What the user gives

| Field | Example | Default |
|---|---|---|
| City | Goa | home city (as an assumption) |
| First day, number of days | Fri 30 Oct, 4 days (1-14) | none, asked |
| Everyday plan | casual, out 10:00 for 8 h, 3 h outdoors, on foot | casual daytime, public transport |
| Extra events | day 2 and 4: party 20:00, 4 h, 0.5 h outdoors, cab | none |
| Travel days | day 1 and the last day as `travel` | on |
| Re-wear limits | wear a top on up to 2 days, a bottom 3, a dress / one-piece 1 | 2 / 3 / 1, asked on the form |

### How it works

1. **Slots.** Every day and event becomes a slot with its own `OutfitRequest`
   (occasion, formality, that slot's weather advice, the profile's taste).
2. **Weather.** One geocode and one forecast fetch for the whole span, cut into one
   `WeatherSummary` per slot. Days past the forecast's 16-day reach use the same dates
   last year from the Open-Meteo archive (`climate.py`), said as an assumption ("the
   forecast isn't out yet: planned on last year's weather for these dates").
3. **Candidates.** `all_outfits` per slot, kept at score >= `GOOD_OUTFIT` (the
   shopping bar).
4. **Pack (greedy cover).** Repeatedly add the outfit that covers the most uncovered
   slots per new piece (shoes and outerwear count as pieces, so one pair that goes
   with everything wins); after each pick, every slot that some outfit made only of
   packed pieces now fits counts as covered. Ties go to the higher score.
5. **Re-wear limits.** The user's limits (`Trip.wears`: tops, bottoms, one-pieces;
   defaults `TOP_WEARS` 2, `BOTTOM_WEARS` 3, `ONE_PIECE_WEARS` 1); shoes and outerwear
   any number. If a limit leaves a slot without an outfit, packing goes on (step 4)
   until it has one; if the wardrobe runs out, the slot reuses the least-worn piece
   and says so once, on the day it first passes the limit ("your navy shirt again:
   worn on more than 2 days from here, nothing else fits").
6. **Days.** Each slot gets an outfit from the packed pieces (no top on two days
   running when there's a choice, then the least-worn pieces, then the best score;
   two slots on one day may share an outfit and count as one day of wear), finished
   with accessories (`finish_outfit`). Only pieces some day uses are packed.
7. **Gaps.** A slot nothing fits is a gap with the closest misses, as in chat
   ("nothing for the party on day 2: no bottom dressy enough").

Output `PackingPlan`: pieces to pack (grouped by category, each with "days 1, 3"),
the plan per day (date, event, weather line, outfit, reasons), gaps, assumptions,
and a line for what isn't in the wardrobe ("+ socks and underwear for 4 days").

### Confirm first (requested 2026-10-25)

"Before starting confirm both the dates and the location, full name, state and
country, and then tell what the weather is like, the occasion etc, and only after
confirmation start the recommendation."

Two steps, and nothing is packed before the user says yes:

1. **Check the trip** (`preview_trip`, `POST /packing/preview`): the city name gives
   up to `PLACE_MATCHES` (5) places, each with its full name, state and country
   ("Panaji, Goa, India" / "Genoa, Liguria, Italy"); the first is preselected and the
   user may pick another or retype the name. For the chosen place it shows:
   - the dates: "Fri 30 Oct - Mon 2 Nov 2026, 4 days";
   - one row per day and event: date, occasion (travel / casual / party ...), time,
     hours away and outdoors, transport;
   - that row's weather: feels-like range, rain chance, wind, UV, sky, and where it
     came from (forecast, or last year on the same dates, or not checked);
   - the re-wear limits and the assumptions.
2. **Pack for this** (`POST /packing`): only with a confirmed place (the one the
   user saw: name, state, country, coordinates, timezone); a trip without one is
   refused (422). The weather is fetched again for that place (free, and nothing the
   user saw can change except a forecast update).

No wardrobe and no DeepSeek call are needed for the check, so it is free and fast.

### Where it lives

- `tools/packing.py` (pure, like `shopping.py`), `scripts/pack_trip.py` (CLI: prints
  the check, asks "Pack for this? [y/N]", `--yes` to skip; `--place 2` picks another
  match).
- API `POST /packing/preview` and `POST /packing` (free: weather only, no DeepSeek
  call).
- UI: a **Packing** page: trip form with the re-wear limits, **Check trip** -> the
  confirmation card (place choice, dates, day rows with weather) -> **Pack for this**
  -> packing list with pictures and day cards. Editing the form or the place goes back
  to the check.
- Not in chat: the page is the only way in (no new intent, intake slots or eval
  scenarios).

### Decisions (2026-10-25)

1. Its own page only, no chat intent.
2. Re-wear limits are asked on the form (prefilled 2 / 3 / 1), not fixed.
3. Luggage limit (max pieces / "carry-on only"): no, just show the count.

### Build steps

| # | Step | Status |
|---|---|---|
| 1 | `tools/packing.py`: `Trip`, `Event`, slots, greedy pack with re-wear limits, day assignment, accessories, gaps; unit tests | done (17 tests; 14 days with 7 parties on the 10-item snapshot: 0.04 s) |
| 2 | Trip weather: one forecast for the span, a summary per slot; last year's archive beyond 16 days; fake-fetch tests | done (`trip_weather`, `plan_trip`, `climate.archive_hours`; live check on Open-Meteo: see notes below) |
| 3 | Confirmation: `weather.find_places` (top matches), `Trip.place` (the confirmed place; no geocoding then), weather source per slot, `preview_trip` -> `TripPreview`; tests | done (`find_places` dedupes name + state + country; `plan_trip` raises `PlaceNotConfirmed` for a city without a place; live: "Manali" offers Tamil Nadu first (feels 34-39 °C) and Himachal Pradesh second (11-18 °C, 3-7 °C at night); a preview for a picked place lists only that place, so the page keeps the first preview's matches) |
| 4 | API `POST /packing/preview` + `POST /packing` (needs a confirmed place) + `scripts/pack_trip.py` (asks before packing) | done (`api/packing.py`: `PlaceOut` carries the label, sent back as `place`; city defaults to the home city, as an assumption; `PlaceNotConfirmed` -> 422; `cli.trip_event` parses `--event DAY,OCCASION,HH:MM,HOURS,OUTDOORS,TRANSPORT`. Live on the real 26-item wardrobe, Manali (Himachal) 3 days + a party: 6 pieces (polo + button-down, grey chinos, hiking shoes, webbing belt by day, black belt for the party); gaps: no warm outerwear, no scarf for the 3-7 °C evening, no party shoes) |
| 5 | Packing page in the UI: form -> confirmation card -> plan | done (`ui/packing_page.py`, client `Trip` / `TripPreview` / `PackingPlan`; the place radio keeps the first check's matches; adding or removing an event, or checking again, throws the check and plan away; 7 AppTest tests, no DeepSeek call) |
| 6 | Docs (`docs/phases/packing.md`, user guide, README, CLAUDE.md), live check on the real wardrobe | done (the page driven headless against the real API: Manali 4 days + 2 parties, 7 pieces, limits kept; see `docs/phases/packing.md`) |
| 7 | From use (2026-11-05): dresses limit 0, formal events, evenings in | done. A one-piece limit of 0 packs none (`Wears.one_pieces` >= 0). Formal got nothing because no bottom reached formality 4 (black trousers tagged 3): the user chose a rule over retagging, so plain bottoms may be one point under a formal/wedding range; the "nothing fits" gap now names the short piece. "Evenings in" (user's choice over lounge events only) adds an indoor lounge slot per night: no weather, shoes or accessories, lounge-tagged bottoms only, with a hint when there are none. Real wardrobe: formal = dress shirt + black trousers + belt; evenings show the lounge gap |
| 8 | From use (2026-11-06): no repeated outfits, and a script to check it | done. `scripts/check_variety.py` (`tools/variety.py`) found the same top + bottom on two days on every trip length, and limits broken on 14 days with 17 pieces unused. Now a base isn't worn on two days unless nothing else fits (with a note), even at the cost of one more piece (assumed from the request, not asked: 2 days now packs 5 pieces, was 4); options keep 30 different bases; `wear_again` passes the fewest limits. Real wardrobe: 2-7 days clean; 14 days only runs out of bottoms |

Notes from step 2 (live check, 2026-10-25, a 5-day trip starting 13 days ahead):

- Slots the forecast has no complete hours for (or a failed forecast) also fall back
  to last year's archive, not just days past 16. The archive has no rain chance or UV:
  a wet hour (>= `WET_HOUR_MM` 0.1 mm) counts as a 100% chance, and the UV is that of
  a typical day of the day's kind (`climate.TYPICAL_DAY`).
- European models leave the forecast's rain chance null after about 10 days while
  temperature, wind and UV are there. `parse_hourly` now reads a missing rain chance
  off that hour's rainfall instead of dropping the hour (chat benefits too).
- Geocoding takes Open-Meteo's top match: "Goa" is Genoa, Italy, and "Panaji" a town
  in Guatemala. The plan carries the place it used, and a name that differs from the
  typed one is an assumption ("took Goa to be Genoa, Liguria, Italy"). Fixed by the
  confirmation step: the user picks among the top matches before anything is packed.

---

## 5. Wear feedback ("I wore this")

**Status:** done, live-checked 2026-10-27 (`docs/phases/wear-feedback.md`; requested 2026-10-25: "put a photo of you wearing the dress it
recommended and use it as feedback"; decisions agreed 2026-10-25)

From any suggested outfit (chat card, packing day) the user presses **I wore this**,
rates it and can add a photo of themselves wearing it. The log then nudges future
suggestions.

### What the user gives

| Field | Values |
|---|---|
| Rating | loved it / fine / not again |
| What was off (any) | too warm, too cold, too formal, too casual, colours didn't go, uncomfortable |
| Note | free text, kept for the user (not used in ranking) |
| Photo | optional, of the user wearing the outfit |

### The photo

- Cut out in `worn` mode (upper / lower, or full) and embedded with CLIP, then
  compared with the outfit's pieces: "your navy shirt: yes; the trousers look like
  your grey chinos, not the beige ones". The user confirms, so the log holds what was
  **actually** worn.
- Kept only on this computer (`data/worn/`, gitignored), re-encoded like the profile
  photo (upright, smaller, no EXIF). **Never sent to DeepSeek**, no LLM call: the
  check is local and free. Nothing judges the person's body or looks.
- The match thresholds (`DUPLICATE` 0.85 / `SIMILAR` 0.75) were calibrated on
  re-shots of the same garment; a worn photo against a flat cutout may score lower,
  so they are re-calibrated on the user's real photos before use.
- A garment in the photo that isn't in the wardrobe is reported ("not in your
  wardrobe"), not added.

### What the feedback changes (ranking only, never a ban)

Same mechanism as taste: named constants, a reason on the outfit for each nudge.

| From the log | Effect |
|---|---|
| loved it | that top + bottom (or one-piece + shoes) pairing +`LIKED_PAIR` ("you loved this last time") |
| not again, or colours didn't go | that pairing −`DISLIKED_PAIR` |
| worn in the last `RECENT_DAYS` days | each piece −`WORN_RECENTLY` (variety) |
| too cold / too warm | after `WARMTH_REPORTS` more of one than the other, the warmth range shifts by 1 for that user (an assumption in the reply: "you've found outfits too cold before") |
| too formal / too casual, uncomfortable | logged and shown, not used yet (too little data to tune a rule) |

### Storage and where it lives

- `outfit/wear_log.py`: `WearEntry` (date, occasion, piece ids, rating, issues, note,
  photo, felt temperature range, source: chat thread or packing), `WearLog` store in
  `data/wear_log.json` (temp file swapped in, in memory for tests, like
  `ProfileStore`). Deleting a garment leaves its entries, shown as "(removed)".
- API: `POST /worn`, `GET /worn`, `DELETE /worn/{id}`, `PUT /worn/{id}/photo` (match
  report back), `GET /worn/{id}/photo`.
- Ranking: `OutfitRequest` gains liked / disliked pairs, recent pieces and a warmth
  bias, filled by orchestrate (and packing) from the log, like `taste_fields()`.
- UI: **I wore this** on chat outfit cards and packing days; a **History** page (photo,
  pieces, rating, delete).

### Decisions (2026-10-25)

1. The photo stays local: CLIP check only, no DeepSeek comment.
2. A rating is always required; the photo is optional.
3. Starting values (to tune on the real wardrobe): `LIKED_PAIR` 0.05, `DISLIKED_PAIR`
   0.1, `WORN_RECENTLY` 0.05 over `RECENT_DAYS` 7, `WARMTH_REPORTS` 2.

### Decision (2026-10-26): DeepSeek may look at the photo, opt-in

Requested: "have DeepSeek have a look at the photo ... and give them score on various
metrics". This replaces decision 1 for this one use: the matching stays local, and
the photo leaves the machine only when the user presses **Ask DeepSeek about this
look** (one call per photo, a re-encoded copy without EXIF, like the colouring
analysis). It scores **the outfit as worn, never the person**: no body, attractiveness,
age, gender or ethnicity, the same rule as `colouring.md`. Agreed metrics, each 1-5
with a reason: fit, length and proportion, colour on you (uses the profile's colouring
when there is one), coordination, right for the occasion, overall polish; plus one or
two tips. No person in the photo -> refused, nothing saved. The scores are kept with
the wear-log entry and shown in History; they don't change the rating or the ranking.

### Build steps

| # | Step | Status |
|---|---|---|
| 1 | `WearEntry` / `WearLog` store + API (`/worn`), tests | done (`outfit/wear_log.py`: `Rating`, `Issue`, `WearEntry` (pieces deduped, too warm + too cold refused), `WearLogStore` (newest first, temp file swapped in); `api/worn.py`: `POST /worn` (201; pieces must be in the wardrobe, no future date; default today), `GET /worn?limit=`, `DELETE /worn/{id}`; deleted garments counted as `removed`; `BadWearLog` -> 422; 14 tests) |
| 2 | Feedback in ranking (pairs, recent wear, warmth bias) via `OutfitRequest`, reasons, tests | done (`tools/feedback.py`: `feedback(entries, items, today)` -> `Feedback` (latest rating of a pairing wins; colours issue = disliked; deleted pieces skipped), `shifted(advice, bias)`; `outfits.pairing` + `history()` in the score; chat via `Context.wear_log`, packing via `plan_trip(history=)` without recent wear; the warmth note is an assumption only when there is a forecast; 18 tests) |
| 3 | Photo: worn cutouts + CLIP match against the outfit; calibrate on the user's photos (asks for them) | done (`Pipeline.worn_vectors` (segment + embed, no tag, nothing saved); `tools/worn_match.py`: each cutout vs the wardrobe pieces of its region's kind -> `Verdict` yours / unsure (another piece beats the logged one by > `CLEAR_WIN` 0.03) / extra / unknown (best < `NOT_IN_WARDROBE` 0.70), `unchecked` = logged pieces no cutout shows; `PhotoMatch` kept on the entry; `PUT /worn/{id}/photo` (checked first, kept in `data/worn/` re-encoded without EXIF only if a garment was found), `GET`/`DELETE .../photo`, `PATCH /worn/{id}` {pieces} re-checks the photo; 17 tests. Values from 4 real photos (right piece 0.73-0.82, wrong up to 0.79, dark trousers 0.01 apart); the user chose to tune them later) |
| 3b | Look scores: `prompts/look.md`, one vision call -> `LookScores` (6 metrics + tips) kept on the entry, `POST /worn/{id}/look`; refused without a person | done (`outfit/look.py`: `score_look(photo, pieces, occasion, colouring, client, settings)`, a 768 px copy without EXIF + the logged pieces, occasion and colouring summary to `tagger_model` (`tagger_max_tokens`); `LookReply` needs all 6 scores (1-5, a reason each) when `person_visible`, else retried; `person_visible: false` -> `NoPersonVisible` 422, nothing saved; at most 2 tips; a new photo clears the scores; 7 tests. Not live-checked: needs a paid run) |
| 4 | UI: **I wore this** on outfit cards (chat, packing), History page (photo, match, look scores) | done (`ui/worn_form.py`: `wore_this` expander + form (pieces, rating required, issues, date, occasion (packing: the slot's), note, optional photo; a failed photo still logs the entry); `ui/history_page.py`: entries newest first, photo, match lines with "Use / Add your X" (PATCH), not-checked pieces, "Ask DeepSeek about this look" -> scores table + tips, remove photo, delete with confirm; client models mirror `WornOut` / `wear_log` (tested); 8 tests) |
| 5 | Docs (`docs/phases/wear-feedback.md`, user guide, README, CLAUDE.md), live check | done (live 2026-10-27 over the real API in-process, log in a scratch folder: the 4 photos all matched the logged pieces (0.746-0.844), 68 s first photo then ~6 s; one look call on 21.09: 11 s, outfit-only reasons) |

### Order

Packing first (§4 steps 1-4), then wear feedback, so "I wore this" works on packing
days too.
