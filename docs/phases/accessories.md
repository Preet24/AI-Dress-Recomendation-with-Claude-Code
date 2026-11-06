# Accessories

Belts, ties, bags, sunglasses, hats, scarves, dupattas, watches and jewellery finish an
outfit the planner has already chosen. They are plain rules in `tools/accessories.py`,
like every other outfit rule: no LLM decides which belt goes on.

## The rule that shapes everything: accessories come last

An outfit is chosen first (`suggest_outfits`), then finished (`finish_plan`): each
outfit gets at most one accessory of each kind, appended after its pieces, with **its
score unchanged**. So owning a nice watch never changes *which* outfits are suggested,
only what they are finished with.

```
suggest_outfits(items, request) ──► ranked, varied outfits (clothes, shoes, outerwear)
                                         │
finish_plan(plan, items, request) ──► + one accessory per kind, its reasons, its gaps
```

Orchestrate and `scripts/suggest_outfits.py` run both steps. `all_outfits`, which the
shopping tools count, never sees accessories.

## What kind of accessory

The kind comes from the subtype the tagger wrote ("leather belt", "silk tie", "tote
bag"); the last word that names a kind wins, so a "belt bag" is a bag. There is no new
`ClothingItem` field, so stored items and the tagger stay as they were. Only items in
the `accessory` category have a kind; words like "keychain" match nothing.

## When each one goes on

| Kind | Only when |
|---|---|
| Belt | the bottom is trousers, jeans, chinos, slacks or shorts (not track pants, joggers, leggings, ethnic bottoms or a dress) |
| Tie | the top is a shirt (the word "shirt": not a t-shirt, polo or sweatshirt) and the occasion's formality reaches 4 (office, party, festive, formal, wedding) |
| Dupatta | the outfit has an ethnic top or bottom |
| Sunglasses | strong sun (UV ≥ 6 with an hour or more outside) or riding |
| Hat | strong sun, or felt temperature below 10 °C |
| Scarf | felt temperature below 10 °C |
| Bag, watch, jewellery | always considered; occasion and formality decide |

For every kind: nothing flowy when the weather or ride rules flowy pieces out (a
dupatta on a scooter), and no hat or scarf more than one point warmer than the
weather's range (no wool beanie at 32 °C). The weather advice says `sunny` and
`riding` for this (`WeatherAdvice`).

## Which one wins

Score = half color harmony with the pieces it sits next to (a tie with the shirt, a
belt with the bottom, anything else with the whole outfit), half fit, plus taste:

- **fit** is the same as for shoes: −0.3 for an occasion it isn't tagged for, −0.15
  per formality point out of range, fabrics the weather advises against. Sunglasses
  skip the occasion and formality part: they are worn outside and come off indoors
  (the live check gave office sunglasses "too casual" before this);
- **belt and shoes**: +0.1 for the same leather (black with black, brown family with
  brown), −0.2 for black with brown;
- **tie and shirt**: −0.15 for a patterned tie on a patterned shirt;
- **taste**: liked and avoided colors, as for clothes.

The best of each kind is added at a score of 0.6 or more (the same bar as shoes).
A requested accessory ("with my silk tie") is always added; the rules it breaks
become warnings, like any requested piece.

## Missing accessories

Only when the outfit really calls for one, so the gap list doesn't nag:

| Gap | When |
|---|---|
| no sunglasses for the strong sun | strong sun and no sunglasses fit |
| no scarf for the cold | below 10 °C and no scarf fits |
| no tie suits formal / wedding | the occasion starts at formality 4, the outfit has a shirt, and the user owns ties but none fits |

## In the reply and on the cards

Accessories travel as ordinary outfit pieces (category `accessory`, last), so the API
and the UI cards show their pictures with no change. The stylist prompt names them
after the clothes ("finish it with your brown leather belt"); the critic treats
accessories listed in the outfit as part of it, catches an invented "your silver
watch" (accessory words are in its garment word list) and requires the accessory gaps
to be passed on.

## Shopping

- **Should I buy this?** An accessory never makes an outfit possible, so it skips the
  with/without comparison. "New outfits" are the good outfits it would be added to,
  "fixes" are the accessory gaps it closes. It is a **buy** only with a fix
  (sunglasses when strong sun has none), a **maybe** when it just goes with outfits (a
  belt), a **skip** when none calls for it (sunglasses for a cold-only climate).
- **What should I buy?** The missing list includes the accessory gaps of each
  situation's good outfits (`shopping.accessory_gaps`). The catalogue has the three
  accessories that can fill a gap: sunglasses, a wool scarf and a silk tie (men; only
  ever suggested when the user owns ties but none fits). A belt, bag or watch never
  fills a gap, so it is not in the catalogue. At equal gaps fixed, clothes rank before
  accessories: a coat for the cold matters more than sunglasses.
- The typical days the shopping tools plan for now carry their own UV (hot 9, mild 6,
  cold and rainy 3), so only hot and mild days count as strong sun.

On the 10-item eval snapshot (no accessories owned), all four weathers:

| Shop for | Suggestions, in order |
|---|---|
| casual, office, party | navy wool coat, black loafers, black sunglasses, maroon wool scarf |
| wedding, festive, formal (woman) | navy trousers, black wool coat, maroon juttis, maroon wool scarf, black sunglasses |
| sports, office (man) | black sports tee + navy track pants (pair), navy wool coat, brown loafers, navy running shoes |

## Photos

Accessories need a photo of the item alone ("Garment alone" on the Wardrobe page):
worn photos are split only into top and bottom (or one full piece), so a belt or
sunglasses in them are not cut out separately. Prescription glasses are not an outfit
choice; a profile setting for them (frame color in color matching) is left for later.

## Code and tests

| File | Role |
|---|---|
| `tools/accessories.py` | `Kind`, `kind_of`, the rules (`takes_belt`, `is_shirt`, `called_for`, `problems`), `score_accessory`, `choose_accessories`, `finish_outfit`, `finish_plan` |
| `tools/weather.py` | `WeatherAdvice.sunny`, `.riding` |
| `tools/outfits.py` | accessories are add-ons (`ACCESSORY_SLOTS`); a requested accessory doesn't block outfits |
| `tools/shopping.py`, `tools/staples.py`, `tools/staples.toml`, `tools/climate.py` | accessory candidates, accessory gaps, staples, UV per typical day |
| `agents/orchestrate.py`, `agents/critic.py`, `prompts/stylist.md`, `prompts/critic.md` | in the plan, the reply and its checks |

Tests: `test_accessories.py` (kinds, every rule, leather, one per kind, requested,
`finish_plan`, orchestrate), plus accessory cases in `test_critic.py`,
`test_shopping.py`, `test_staples.py` and `test_weather.py`.

## Live check (2026-10-24, 26-item wardrobe, no DeepSeek calls)

Added: a black leather belt (formality 4), a blue webbing belt (2), black sunglasses.

- Belts went only on chinos, jeans and trousers: leather for office and party, webbing
  for casual and rainy days. Sunglasses came only on strong-sun days, never on rainy
  ones. "What should I buy?" dropped its sunglasses gap and suggested a grey wool scarf
  for the cold, after a coat and loafers.
- Two fixes came out of it. **Socks** had been tagged three ways (black socks as a
  bottom, so "shirt + black socks" was an office outfit; off-white socks as footwear;
  beige socks as an accessory): the tagger prompt now says socks, stockings and tights
  are accessories (they have no kind, so outfits ignore them), and the two wrong items
  were set to accessory by a tag edit. **Sunglasses** stopped being judged by occasion
  (above).
- Not checked: ties (formal and wedding have no outfit yet: no bottom is dressy
  enough), and the agent eval after the prompt changes (its wardrobe snapshot has no
  accessories, so it would only show that nothing regressed).
