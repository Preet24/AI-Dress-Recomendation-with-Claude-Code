# Wear feedback: "I wore this"

Under any suggested outfit (a chat card or a packing day) the user presses **I wore
this**, rates it and can add a photo of themselves wearing it. The log then nudges
future suggestions, the photo is checked against the wardrobe on this computer, and,
only if the user asks, DeepSeek scores the outfit as worn. Planning and decisions are
in [`../features_to_add.md`](../features_to_add.md) §5.

```
I wore this ─► POST /worn ─► wear_log.json ─► tools/feedback.py ─► OutfitRequest
   (rating,                                     liked / disliked pairs,    (chat: orchestrate,
    issues,                                      recent pieces,              packing: plan_trip)
    photo?)                                      warmth bias
      │
      └─► PUT /worn/{id}/photo ─► worn segmentation + CLIP (local) ─► tools/worn_match.py
                                                                       └─► PhotoMatch on the entry
                                    "Ask DeepSeek about this look" ─► POST /worn/{id}/look
                                                                       └─► LookScores on the entry
```

## The log

`outfit/wear_log.py`: a `WearEntry` holds the date, occasion, the pieces actually
worn, a **required** rating (loved / fine / not again), any issues (too warm, too cold,
too formal, too casual, colours didn't go, uncomfortable), a note, the source (chat,
packing, other), and, once there is a photo, the match report and look scores.
`WearLogStore` keeps it in `data/wear_log.json` (written to a temp file and swapped in,
like the profile; in memory for tests) and photos in `data/worn/<entry id>.jpg`. A
garment deleted later stays in its entries; the API reports it as `removed`.

## What it changes in the ranking

`tools/feedback.py` turns the log into the same kind of adjustments as taste: named
constants, a reason on the outfit for each, never a ban.

| From the log | Effect |
|---|---|
| a pairing (top + bottom, or one-piece + shoes) last rated **loved** | +`LIKED_PAIR` 0.05, "you loved your X with Y last time" |
| last rated **not again**, or "colours didn't go" | −`DISLIKED_PAIR` 0.1 |
| a base garment worn in the last `RECENT_DAYS` (7) | −`WORN_RECENTLY` 0.05 each (chat only; packing plans days that haven't happened) |
| too cold − too warm ≥ `WARMTH_REPORTS` (2), or the reverse | the weather's warmth range (and outerwear minimum) moves one point, with an assumption saying why |

Only a pairing's **latest** entry counts, so "fine" after "loved" clears it. The
warmth note appears only when there is a forecast to shift. Too formal / too casual
and uncomfortable are logged and shown but not used: too little data to tune a rule.

## The photo check (local, free)

`Pipeline.worn_vectors(photo)` runs the same `worn` segmentation as ingestion
(`u2net_cloth_seg`: upper / lower, or one full garment) and CLIP on each cutout, but
tags nothing and saves nothing. `tools/worn_match.py` then compares each cutout only
with wardrobe pieces of its kind (upper: tops, ethnic tops, outerwear; lower: bottoms;
full: one-pieces) and with what the user logged:

| Verdict | When | Shown as |
|---|---|---|
| yours | the logged piece is the best match, or within `CLEAR_WIN` (0.03) of it | ✅ "your tan chinos: yes" |
| unsure | another piece beats the logged one by more than 0.03 | ❓ "looks more like your grey chinos than your navy trousers: which was it?" + **Use your grey chinos** |
| extra | it matches a piece that isn't in the entry | ➕ + **Add your ...** |
| unknown | the best match is under `NOT_IN_WARDROBE` (0.70) | ⚠️ "may not be in your wardrobe" |

Logged pieces no cutout can show (shoes, accessories, a region not found) are listed
as **not checked**. The fix buttons send `PATCH /worn/{id}` with the corrected pieces,
and the photo is checked again. The photo is checked before it is kept: one with no
garment in it is refused (422) and nothing is saved.

### Why no score decides alone

The shopping thresholds (`DUPLICATE` 0.85 / `SIMILAR` 0.75) came from re-shots of a
garment lying flat. A person in a room photo is small, lit differently and partly
hidden, so scores are lower and closer. On the user's four 26 Oct photos:

| Photo | Right piece | Next best |
|---|---|---|
| turquoise polo | 0.82 | black polo 0.74 |
| brown flannel shirt | 0.76 | blue flannel 0.70 |
| black button-up (21.09) | 0.77 | "balck dress shirt" 0.66 |
| tan chinos (21.09) | 0.82 | black trousers 0.79 |
| black button-up (21.11) | 0.73 | dress shirt 0.66 |
| black trousers (21.11) | 0.80 | grey chinos 0.79 |

Right pieces scored 0.73–0.82 and a wrong one up to 0.79, and two dark trousers were
0.01 apart, so a fixed "it's yours above X" would be wrong both ways. Instead the check
starts from what the user said, doubts it only when another piece is clearly closer,
and the user decides. The flip side: a wrong pick between two similar dark trousers
won't be caught. The values are starting points, to tune on more photos.

### Live check (2026-10-27)

The real API in-process (real wardrobe, rembg and CLIP on CPU), the log in a scratch
folder, each photo logged with the pieces above:

| Photo | Result |
|---|---|
| 20.49 polo | ✅ turquoise polo shirt (0.844) |
| 21.03 flannel | ✅ brown flannel shirt (0.776) |
| 21.09 shirt + chinos | ✅ black button-up shirt (0.778), ✅ tan chinos (0.784) |
| 21.11 shirt + trousers | ✅ black button-up shirt (0.746), ✅ black trousers (0.815) |

68 s for the first photo (loading rembg and CLIP), then about 6 s each. Scores differ
a little from the calibration run because the check sees the kept 1024 px copy.

## Look scores (opt-in, one DeepSeek call)

`outfit/look.py` → `score_look(photo, pieces, occasion, colouring, client, settings)`
sends a 768 px copy without EXIF, the pieces the user logged, the occasion and the
profile's colouring summary (if any) to `tagger_model`. The prompt
(`prompts/look.md`) asks for six scores from 1 to 5, each with a reason that names a
garment or detail: **fit, proportion, colour on you, coordination, right for the
occasion, polish**, plus one or two tips. It scores the outfit as worn, **never the
person**: no body, shape, face, attractiveness, age, gender, ethnicity or health, the
same rule as the colouring analysis. `LookReply` requires all six scores when
`person_visible` is true (else one retry with the error); `person_visible: false`
raises `NoPersonVisible` (422) and nothing is saved. The scores are kept on the entry
and shown in History; they change neither the rating nor the ranking, and a new photo
clears them.

Live, on the 21.09 photo (11 s, one call): fit 4 ("the chinos are a touch long and
stack at the ankle"), proportion 4, colour 4, coordination 3 and occasion 3 ("the
flip-flops drop the formality"), polish 2 ("flip-flops and the crumpled trouser hem
undercut the finish"); tips: swap the flip-flops for loafers or white sneakers, cuff or
hem the chinos. Every reason was about garments; the colour reason used the profile's
colouring.

## API and UI

| Endpoint | Does |
|---|---|
| `POST /worn` | Log an outfit: pieces in the wardrobe, no future date (default: today), rating required |
| `GET /worn` | Entries newest first, with pieces, `removed`, `photo_url`, `match`, `look` |
| `PATCH /worn/{id}` | Set the pieces actually worn; re-checks the photo |
| `DELETE /worn/{id}` | Remove an entry and its photo |
| `PUT / GET / DELETE /worn/{id}/photo` | Add (check first, then keep), view, remove the photo (the entry stays) |
| `POST /worn/{id}/look` | The look scores (1 DeepSeek call) |

- **I wore this** (`ui/worn_form.py`) is an expander with a form under every chat
  outfit and packing day: the pieces (all ticked), rating, issues, date and occasion
  (a packing day fills in its own), note, optional photo. A failed photo still logs the
  entry and says why it wasn't kept.
- **History** (`ui/history_page.py`): each entry with its photo, the match lines and
  fix buttons, **Ask DeepSeek about this look** (with a note that it sends the photo),
  the scores table and tips, remove photo, delete with a confirmation.

## Tests

- `tests/test_wear_log.py`, `tests/test_api_worn.py`: entries, the store (photos on
  disk and in memory, no EXIF), every endpoint, the photo check over a fake
  segmenter + CLIP (a photo with no garment is not kept), corrections, look scores.
- `tests/test_feedback.py`: pairings, latest rating wins, recent wear, the warmth
  bias and its note, chat and packing.
- `tests/test_worn_match.py`: every verdict, the region's kind, unchecked pieces.
- `tests/test_look.py`: what is sent (small copy, no EXIF, pieces, occasion,
  colouring), no person, a missing or out-of-range score retried.
- `tests/test_ui_history_page.py` (+ a packing page check): the client mirrors, the
  form, the History page's fixes, look button and delete, headless against the real
  API over fakes.
