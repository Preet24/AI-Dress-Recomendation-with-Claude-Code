# Shopping assistant

Two tabs on the Shopping page, sharing one scoring module (`tools/shopping.py`):

- **What should I buy?** looks at the wardrobe and the profile and says what is
  missing for the chosen occasions, and which common pieces would help most. Local
  rules only: no DeepSeek call.
- **Should I buy this?** checks a photographed garment (below, from "Flow").

## What should I buy?

```
occasions ─► situations (saved places for them, at their dress code)
home city ─► typical weather kinds
wardrobe ──► baseline: every outfit per situation × weather ──► what's missing
                         │
staples.toml (33 pieces incl. 3 accessories, gender-filtered)
   └─ per staple: color with most "goes with" partners ──► check_candidate ──► gain
                         │
   greedy: take the best "buy", add it to the wardrobe, score the rest again (≤ 5)
```

- **What's missing**: situations with no outfit scoring ≥ 0.7, with the planner's
  closest misses ("wedding in any weather (bottoms: lavender skirt has formality 3,
  wanted 4-5 ...)"), then its gaps ("no footwear suits office", in which weathers),
  including the accessory gaps of the good outfits ("no sunglasses for the strong
  sun", "no scarf for the cold"; see [`accessories.md`](accessories.md)).
- **The staples** are tag sets, not photos: `src/outfit/tools/staples.toml` (category,
  subtype, usual colors, fabric, formality, warmth, occasions; `gender` for pieces only
  one gender usually shops for). Edit the file to change the catalogue; rows are
  validated on load.
- **Color**: colors the user avoids or should go easy on are skipped, and so are
  colors in which they already own that kind of piece ("white sneakers" vs their
  "white canvas sneakers"). Of the rest, the one that goes with most of their pieces
  wins (liked colors break ties), and the card says why: "navy goes with 9 of your 9
  tops, bottoms and one-pieces".
- **Ranking**: unlocked situations, then gaps fixed, then clothes before accessories
  (a coat for the cold matters more than sunglasses), then new outfits, then how many
  of the occasions the piece is made for (running shoes beat sneakers for sports).
- **Accessories** in the catalogue are only those that can fill a gap: sunglasses, a
  wool scarf, a silk tie (men). Like shoes and outerwear, they are suggested only for
  a gap.
- **Greedy**, so suggestions don't overlap: after black loafers fill "no footwear
  suits office", other shoes no longer fix it and drop out. Shoes and outerwear need
  an unlock or a gap fixed: a new pair of shoes makes every outfit "new", so its
  outfit count says little.
- **Pairs**: some situations need two new pieces (no sports top *and* no sports
  bottom). When no single staple unlocks anything but a situation has no outfit, every
  top × bottom staple pair is tried together, and the best is suggested as a pair, each
  scored with the other owned.

On the 10-item eval snapshot with no home city (all four weathers):

| Shop for | Suggestions, in order |
|---|---|
| casual, office, party | navy wool coat (cold days), black loafers, black sunglasses, maroon wool scarf |
| wedding, festive, formal (woman) | navy trousers (unlock wedding and formal), black wool coat, maroon juttis, maroon wool scarf, black sunglasses |
| sports, office (man) | black sports tee + navy track pants (pair), navy wool coat, brown loafers, navy running shoes |

It takes about 1-3 s on a CPU. Most of it used to go into converting the same garment's
LAB color to RGB for every outfit combination; `color_harmony.ryb_hue` is now cached
(9 s -> 1.3 s).

## Should I buy this?

The user photographs something they might buy (a shop photo, a screenshot, a photo in
the store). The app cuts it out and tags it exactly like a wardrobe photo, does **not**
store it, and reports whether it earns a place in the wardrobe. Everything after the
tagging is plain, tested Python: no LLM decides the verdict.

### Flow

```
photo ──► Pipeline.describe_photo ──► CandidateStore (memory, newest 20)
            cut out, tag (DeepSeek),        │
            color, CLIP vector              ▼
                                   tools/shopping.check_candidate
   wardrobe (Qdrant) ─────────────►   for each situation × weather kind:
   look-alikes (CLIP, same category) ►   all_outfits(without) vs all_outfits(with)
   profile: places, occasions, taste ►  goes with / clashes (color harmony)
   home city ► tools/climate ────────►  verdict + reasons
                                                │
                       "Add to wardrobe" ◄──────┘  (stores tags + vector, no re-tag)
```

### What is checked (both tabs)

- **Situations**: saved places first, at their dress code's formality (Office,
  business casual → formality 3-4); then the occasions the user ticked on the
  Shopping page that no place covers; with neither, casual, office and party.
- **Weather**: the home city's last year from Open-Meteo's archive (free, no key).
  Each day is hot (warmest hour feels ≥ 32 °C), cold (< 16 °C), rainy (≥ 5 mm) or
  mild; kinds on at least 8% of days count. Pune: hot 48%, mild 41%, rainy 11%, so
  a winter coat is never "needed" there. Each kind becomes advice through the same
  `advise()` rules as a real forecast, for a typical daytime outing (UV 9 on hot days,
  6 mild, 3 cold and rainy, so only hot and mild days are "strong sun"). Looked up
  once per city and day.

### The report

| Part | How |
|---|---|
| Goes with / clashes | color harmony against the pieces it is worn with (≥ 0.75 / < 0.55) |
| New outfits | distinct outfits that include it and score ≥ 0.7 (other shoes, other situations count once) |
| Unlocks | situations that had no good outfit before, grouped ("party in any weather") |
| Fixes | wardrobe gaps it closes ("no footwear suits party") |
| Closest piece you own | CLIP similarity, same category |
| Verdict | skip if no new outfit or a near-duplicate; buy if it unlocks, fixes or adds ≥ 6 outfits; maybe otherwise |
| Accessories | no with/without comparison (they never change the outfits): new outfits = good outfits it would be added to, fixes = accessory gaps it closes; buy only with a fix |

The planner's `suggest_outfits` caps how often one garment repeats (for variety in
chat), which would hide counts, so the check uses `all_outfits`, the uncapped list.

### Calibrating "near-duplicate"

Same-category CLIP similarity on the real 20-item wardrobe:

| Pair | Similarity |
|---|---|
| Same garment, mirrored / darker / warmer light | 0.91-0.99 (median 0.97-0.98) |
| Same garment, rotated and cropped | 0.82-0.99 (median 0.94) |
| Different garments, most alike (tan chinos vs black trousers) | 0.79 |
| Similar shirts (two flannels, two polos) | 0.70-0.78 |

`DUPLICATE = 0.85` sits in the gap; 0.75-0.85 is mentioned ("looks similar to your
...") without changing the verdict. A real shop photo differs more than these edits,
so the threshold errs towards missing a duplicate rather than flagging a new piece.

## Cost and privacy

"What should I buy?" is free and sends nothing but the city name (to Open-Meteo).
"Should I buy this?": one DeepSeek vision call per garment found in the photo (the cutout only, as for the
wardrobe), plus 2-26 s of CPU segmentation. The climate lookup sends the city name to
Open-Meteo. Checked photos and cutouts wait in `data/candidates/` (gitignored) until
added or pushed out by newer checks.

## Code

| File | Role |
|---|---|
| `ingestion/pipeline.py` | `describe_photo`: everything but storing |
| `ingestion/candidates.py` | `CandidateStore`: put, get, add_to_wardrobe |
| `tools/climate.py` | `typical_kinds`, `typical_advice` |
| `tools/shopping.py` | `baseline`, `check_candidate`, `verdict`, `group_where`, thresholds; `accessory_effect`, `accessory_gaps` |
| `tools/staples.py`, `tools/staples.toml` | the catalogue, `pick_color`, `missing`, `suggest_purchases`, `best_pair` |
| `api/shopping.py` | `POST /shopping/suggest`, `POST /shopping/check`, candidate image, add |
| `ui/shopping_page.py` | occasions, the two tabs, suggestion and report cards |
