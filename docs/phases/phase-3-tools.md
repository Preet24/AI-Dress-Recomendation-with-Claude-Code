# Phase 3 — Tools

Four deterministic, unit-tested tools that hold all the styling knowledge (more were
added on top of them later, marked below). The Phase 4
agents call them; the agents decide *what* to do, these tools decide *what is true*.
Every rule is a named constant, so every outcome can be explained and tested.

| Tool | File | In → out |
|---|---|---|
| Wardrobe search | `tools/search.py` | text or an item → ranked items |
| Weather | `tools/weather.py` | place + outing → forecast → clothing advice |
| Color harmony | `tools/color_harmony.py` | two or more items → score 0..1 + reason |
| Outfit assembly | `tools/outfits.py` | wardrobe + request → ranked, varied outfits + gaps |
| Item matching (added in Phase 5) | `tools/match.py` | "my lavender skirt" → the items whose tags fit |
| Typical weather (added later) | `tools/climate.py` | home city → which of hot / mild / cold / rainy days it has |
| Shopping (added later) | `tools/shopping.py`, `tools/staples.py` | a garment you might buy → buy / maybe / skip; the wardrobe → what to buy |
| Accessories (added later) | `tools/accessories.py` | a chosen outfit + your accessories → at most one belt, tie, bag ... each, and the missing ones |

## 1. Wardrobe search

`WardrobeSearch.by_text(query, filters)` and `.similar_to(item_id, filters)`.

- Text is embedded with CLIP as `"a photo of {query}"`. On the real wardrobe this
  template ranked the right item first for 8 of 8 test queries; a longer one shrank
  the margin to the runner-up, so it was dropped.
- `similar_to` uses the item's stored vector and leaves the item itself out.
- `ItemFilter` gives the hard constraints; CLIP similarity only ranks.

CLIP similarity means **"looks alike", not "goes together"**: garments cut from the
same photo score high because they share its lighting. So pairing never uses
`similar_to`; it uses color harmony and tags.

## 2. Weather

`weather_for_outing(place, Outing)` → `(Place, WeatherSummary, WeatherAdvice)`.

**Input.** An `Outing`: start time, hours away, hours outdoors, and a `Transport`
(walk, bicycle, two-wheeler, public transit, car).

**Fetching.** Open-Meteo (no API key) over the standard library's `urllib`:
geocoding, then the hourly forecast for exactly the outing's hours (temperature,
feels-like, humidity, rain chance, rainfall, UV, wind, gusts, sky). Server and network
errors are retried with backoff. Times are naive local time at the place. Tests inject
a fake `fetch`.

**Advice.** `advise(summary, outing)` turns the forecast into constraints:

| Rule | Logic |
|---|---|
| Temperature | Uses the **minimum feels-like** over the outing. Riding below 25 °C subtracts wind chill (bicycle 2 °C, two-wheeler 5 °C) |
| Warmth bands | ≥ 26 °C → warmth 1–2 · ≥ 18 → 1–3 · ≥ 10 → 2–4 + outerwear 3+ · ≥ 0 → 2–5 + outerwear 4+ · below → 3–5 + outerwear 5 |
| Exposure | Walking, riding or transit, or ≥ 0.5 h outdoors, counts as exposed. A car with less outdoor time is not: rain, wind and UV only add notes, and outerwear becomes optional |
| Rain | ≥ 50% chance or ≥ 1 mm: avoid suede, silk, satin, velvet, leather; umbrella note; on a bike, a rain jacket is required |
| Wind | Sustained ≥ 25 km/h or gusts ≥ 40 km/h while exposed: no flowy pieces. Riding always rules out flowy pieces (they catch in wheels) |
| Humidity | ≥ 70% and ≥ 25 °C: prefer cotton and linen, avoid synthetic, wool, leather, velvet |
| Sun | UV ≥ 6 with an hour or more outside: light colors, sunglasses, sleeves; light colors also when ≥ 30 °C and exposed |

`WeatherAdvice.item_filter()` gives the **hard** filters (warmth range, flowy) for base
garments. Fabrics, colors and outerwear only adjust ranking.

## 3. Color harmony

`pair_harmony(swatch(a), swatch(b))` and `outfit_harmony(items)` → score 0..1, a
`Relation` and a reason the stylist can quote ("neutral black lets the lavender stand
out, good light-dark contrast").

Three decisions came from measuring the real wardrobe:

- **Hue on the artist's RYB wheel, not LAB.** Styling complements (red–green,
  blue–orange, yellow–purple) are only ~96–114° apart as LAB hue angles. The hue is
  converted LAB → RGB → HSV → RYB, where they are 180° apart.
- **Neutral by name first.** Real photos are very desaturated (the most colorful item
  measured chroma ≈ 19; pastels ≈ 9–12), so a chroma threshold alone would call
  lavender "grey". A garment is neutral if its color *name* contains a neutral word
  (black, white, beige, navy, denim…) or its fabric is denim; chroma < 6 is only the
  fallback.
- **Trust hue only as far as color allows.** Below chroma 10 the measured hue is
  noise, so two such colors score as "soft colors"; hue rules apply fully from chroma
  30.

Scoring: neutral + color 0.85, two neutrals 0.75, colored pairs by hue distance
(analogous and complementary good, ~90° the weakest). Light-dark contrast earns up to
+0.1; two loud non-analogous colors and more than two colors per outfit cost points.

## 4. Outfit assembly

`suggest_outfits(items, OutfitRequest)` → `OutfitPlan` (ranked `Outfit`s + `gaps`).

**Shape of an outfit.** A base (top or ethnic top + bottom or ethnic bottom, or one
one-piece), plus footwear, plus outerwear when the weather calls for it.

**Hard rules, for base garments only:** tagged for the occasion, formality in range
(default per occasion: casual 1–3, office 3–4, party 3–5, wedding 4–5…), and the
weather's warmth range and no-flowy rule.

Two rules are loosened, because a strict version left real requests with nothing (a
casual party in 33 °C Bangalore found no outfit in an 18-item wardrobe):

- **Near occasions.** The tagger gives each garment only a few occasions; only 3 of
  the 18 were tagged `party`. A garment tagged for a near occasion (`NEAR_OCCASIONS`:
  for a party, casual / office / formal) passes when its formality is in range, at
  0.1 fit less, so garments actually tagged for the occasion rank first. A gym top
  never passes for the office.
- **Warm bottoms.** The tagger rates almost every full-length bottom (jeans, chinos,
  trousers) warmth 3, which the heat band (1–2) ruled out. Bottoms may be one point
  warmer than the range (`BOTTOM_WARMTH_SLACK`), at 0.1 fit less and with a note
  ("black trousers is a bit warm for this weather"). Tops get no slack.

When no base fits, the gap names the two closest garments per empty slot and the rules
they broke ("bottoms: black trousers has warmth 4, the weather calls for 1-2"), so the
reply can say what is actually missing.

**Add-ons are scored, not filtered.** Wardrobes have few shoes and jackets, and an
outfit with imperfect shoes beats no outfit. Footwear and outerwear lose fit for a
missed occasion or formality, and are added only at fit ≥ 0.6; otherwise the plan
reports a gap ("no footwear suits party").

**Score** = 0.5 × color harmony + 0.5 × fit. Fit is the mean piece fit (fabric for the
weather, light colors in heat) minus outfit penalties (two busy patterns, pieces whose
formality differs by more than 2).

**Taste** (added later, from the user profile) is a third, uncapped term: per piece,
+0.03 for a main color the user likes, −0.08 for one they avoid (it sinks the outfit
but never bans it), +0.02 for a top or bottom in their preferred fit. It sits outside
fit because piece fit is capped at 1.0 and nearly every piece is already there, so a
bonus inside it would change nothing. Each adjustment adds a reason the stylist can
quote ("yellow t-shirt is in yellow, a color you usually avoid"). The optional photo
colouring analysis adds a weaker +0.02 for colors that suit the user and −0.03 for
colors to go easy on, skipped where their own likes and avoids already matched.

`all_outfits(items, request)` (added for shopping) returns every passing outfit, best
first, without the variety cap and limit, so the shopping tools can count outfits.

**What the user asks for wins.** `must_include` items skip the hard rules but lose
0.15 fit per rule they break, and each break becomes a warning ("medium blue jeans has
formality 2, wanted 3-5"), so the reply can be honest about it.

**Variety.** A base garment appears in at most 2 suggestions; shoes and outerwear
may repeat.

Every outfit carries its `reasons` and `warnings`. These are the facts the Phase 4
stylist explains, and the only ones it may use.

## 5. Item matching (added in Phase 5)

`best_matches(reference, items)` turns what the user calls a garment ("my lavender
skirt", "the white tee") into items by their **tags**: the reference's last word must
be one of the item's garment words (subtype or category), so a color alone never
matches ("blue sweater" is not the blue jeans); then the item whose tags contain the
most of the reference's words wins. Plurals are ignored, "tee" means "t-shirt", and
"off-white" matches "white".

It exists because the Phase 5 eval showed CLIP alone mis-ranks real items
("lavender skirt" scored the beige trousers higher). Callers use CLIP only to break
ties ("black top": two black tops) and for references no tag matches ("something
sparkly"), with the 0.24 threshold.

## 6. Shopping tools (added later)

Built on the tools above for the shopping assistant; explained in
[`shopping-assistant.md`](shopping-assistant.md).

- `climate.typical_kinds(city, today, fetch)`: last year's daily weather from the
  Open-Meteo archive; each day is hot, mild, cold or rainy, and kinds on at least 8% of
  days count. `typical_advice(kind)` turns one into `WeatherAdvice` through the same
  `advise()` rules as a forecast.
- `shopping.check_candidate(...)`: the outfits with and without a candidate in every
  situation and weather → new outfits, unlocked situations, gaps fixed, what it goes
  with and clashes with, look-alikes, and a verdict with reasons.
- `staples.suggest_purchases(...)`: about 30 common pieces plus three accessories (`staples.toml`), each in its
  best color for the wardrobe, scored by `check_candidate` and picked greedily.

Caching `color_harmony.ryb_hue` (a pure LAB → RYB conversion) made outfit scoring about
7× faster, since every combination asked for the same colors again.

## 7. Accessories (added later)

`accessories.finish_plan(suggest_outfits(items, request), items, request)` adds at most
one accessory of each kind (belt, tie, bag, sunglasses, hat, scarf, dupatta, watch,
jewellery) to every outfit, after its pieces and without changing its score. The kind
comes from the subtype words; hard rules (a belt needs trousers or jeans, a tie a
shirt and a dressy occasion, sunglasses strong sun or a ride) pick the candidates, and
harmony with the pieces it sits next to, fit and taste pick the winner. Missing ones
are gaps only when called for ("no sunglasses for the strong sun"). `WeatherAdvice`
gained `sunny` and `riding` for this. Details in [`accessories.md`](accessories.md).

## Commands

```bash
python scripts/search_wardrobe.py "black dress" --formality 3-5 --category one_piece
python scripts/search_wardrobe.py --like <id prefix>
python scripts/check_weather.py --city Pune --start 18:30 --hours-away 4 --hours-outdoors 1 --transport two_wheeler
python scripts/suggest_outfits.py --occasion party --city Pune --start 19:00 --hours-away 4 --hours-outdoors 0.5 --transport car [--with <id>]
python scripts/suggest_outfits.py --occasion party --like navy --avoid black --fit-tops relaxed   # taste, as the profile sets it
```

## Testing

- All the tools are pure functions over `ClothingItem`s or injected dependencies:
  no network, no real store, no CLIP in the default test run.
- Weather tests feed Open-Meteo-shaped JSON through a fake `fetch` and check each rule:
  retries, the 16-day limit, warmth bands, wind chill only when not hot, rain by car vs
  riding, UV only with real time outdoors.
- Color tests pin the decisions above: RYB hues, complements 180° apart, neutral by
  name, then fabric, then chroma, and washed-out colors ignoring their hue.
- Files: `test_search.py`, `test_weather.py`, `test_color_harmony.py`, `test_outfits.py`,
  `test_match.py` (including 12 references on the real wardrobe snapshot); later
  `test_outfit_taste.py`, `test_shopping.py` (typical weather and candidate reports)
  and `test_staples.py` (color choice, greedy picks, pairs), and
  `test_accessories.py` (kinds, every rule, leather matching, `finish_plan`).
