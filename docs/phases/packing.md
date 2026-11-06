# Trip packing list

Describe a trip ("Manali, 4 days from Friday, a party on day 2 and day 4") and get the
**fewest pieces that cover every day**, plus which outfit to wear when. Everything is
plain Python on top of the outfit planner and the weather tools: no LLM decides what
goes in the bag, and no DeepSeek call is made.

## Confirm first

Nothing is packed until the user has seen and accepted where, when and what the weather
will be. The place name alone is not enough: Open-Meteo's search ranks by population,
so "Goa" is Genoa, Italy, and "Manali" is a town in Tamil Nadu (34-39 °C) before the
hill station in Himachal Pradesh (11-18 °C by day, 3-7 °C at night).

```
trip form ──► POST /packing/preview ──► "Is this right?"        ──► POST /packing
              (preview_trip)              places: name, state,       (plan_trip, only
              no wardrobe, free           country (pick one)         with the place
                                          dates, every day and       the user saw)
                                          event with its weather
```

`plan_trip` refuses a trip that has a city but no confirmed place
(`PlaceNotConfirmed`, a 422 from the API), so the check can't be skipped by mistake.

## The trip

| Field | Default |
|---|---|
| City (a town, not a state) | the profile's home city |
| First day, days (1-14) | asked |
| Everyday plan: occasion, leaving time, hours away / outdoors, transport | casual, 10:00, 8 h, 3 h out, public transport |
| Events: day, occasion, time, hours, transport | none |
| First and last day as travel | yes |
| Evenings in (relax / sleep) | no |
| Re-wear limits, in days: tops, bottoms, one-pieces | 2 / 3 / 1, set on the form; one-pieces may be 0 (pack none) |

Each day's everyday plan and each event becomes a **slot** with its own occasion,
outing and weather (`trip_slots`). **Evenings in** add a `lounge` slot at 21:00 on
every day but the last (a 3-day trip has 2 nights), marked `indoors`: it gets no
weather (source "indoors", nothing fetched), no shoes and no accessories, and only
bottoms tagged `lounge` count (jeans or chinos tagged casual don't; a casual t-shirt
does). With nothing to lounge in, the evening shows the gap with a hint ("joggers,
shorts or pyjamas tagged lounge would do"). A one-piece limit of 0 takes one-pieces
out of the plan entirely, so they're never "worn again" either.

At formal and wedding events, a plain bottom (solid, not jeans / chinos / shorts /
denim) may be one formality point below the range, at a small penalty, because the
tagger rates plain dark trousers 3 from a photo; the top must meet the range. On the
real wardrobe this turned a formal evening from nothing into the black dress shirt +
black trousers + black belt.

## Weather for every slot

`trip_weather` makes one geocode (skipped for a confirmed place), **one** forecast
request covering all slots it reaches, and **one** archive request for the rest:

- slots within the forecast's 16 days use the forecast hours of that slot;
- slots past it, slots the forecast has no complete hours for, or all slots after a
  failed forecast use **the same dates last year** (`climate.archive_hours`). The
  archive has no rain chance or UV: a wet hour (>= 0.1 mm) counts as a 100% chance,
  and the UV is that of a typical day of its kind (hot and mild are sunny);
- every slot then goes through the usual `advise()`, so a party at 20:00 by car gets
  different advice from a day of walking.

Where the weather came from is shown per row (forecast / last year / not checked) and
said as an assumption; a missing forecast is never a gap. One fix came out of the live
check: European models leave the rain chance empty after about 10 days while the
temperature and wind are there, so `parse_hourly` now reads a missing rain chance off
the rainfall instead of dropping the hour (chat outings that far ahead work too).

## Packing: a greedy cover

1. **Options.** For each slot, the planner's good outfits (`all_outfits`, score >=
   0.7, the shopping assistant's bar; the best there is when none is good): the best 30
   different bases, each with all its shoes and outerwear. (A plain top 30 was a few
   bases in every pair of shoes, so limits ran out with tops still in the wardrobe.)
2. **Pack.** Repeatedly add the outfit that dresses the most still-undressed slots
   *per new piece*. Shoes and outerwear count as pieces, so one pair that goes with
   everything beats three. After each pick, every slot that some outfit made only of
   packed pieces can dress counts as dressed.
3. **Re-wear limits and no repeated outfits.** A top, bottom or one-piece is worn on at
   most the user's number of days, and the same base (top + bottom, or one-piece) isn't
   worn on two days, so pieces get mixed differently or one more is packed; shoes,
   outerwear and accessories repeat freely. Two slots on one day may share an outfit and
   count as one day. When the wardrobe runs out, a piece or base is worn again, with a
   note the first time ("your navy shirt again: worn on more than 2 days from here,
   nothing else fits"; "your navy shirt with grey chinos again, as on day 1: ...").
   `wear_again` picks the fewest pieces past their limit, so when bottoms run out a
   fresh top still goes with a bottom worn once more.
4. **Days.** Each slot gets an outfit made only of packed pieces: no top two days
   running when there's a choice, then the least-worn pieces, then the best score.
   Accessories finish it (`finish_outfit`), as in chat.
5. **List.** Only pieces some day uses are packed, grouped tops, bottoms, one-pieces,
   outerwear, shoes, accessories, each with its days; plus "socks and underwear for N
   days". Gaps name their days: "no footwear suits party (days 2 and 4)".

On the real 26-item wardrobe, the worst case (14 days, 6 parties, evenings in: 33 slots)
packs in 1.6 s.

## Checking for repeats

`tools/variety.py` (`check_variety(plan, wears)`) and `scripts/check_variety.py` look
for the same base on two days and pieces past their limit (problems), and the same top
two days running (a note), ignoring shoes, outerwear and accessories. The script packs
trips of 2, 3, 5, 7 and 14 days on the wardrobe (rules only: no weather, no DeepSeek) and
exits 1 on a problem. It is what found the repeats and the option cap above. Real
wardrobe after the fix (2026-11-06): 2-7 days clean, with one more piece on short trips
(2 days: 5 pieces, was 4); 14 days reports only that two bottoms go past 3 days, which is
true: 4 casual bottoms at 3 days each cover 12 days.

## Where it lives

| File | Role |
|---|---|
| `tools/packing.py` | `Trip`, `Activity`, `Event`, `Wears`, `trip_slots`, `pack` (pure), `trip_weather`, `preview_trip`, `plan_trip`, `weather_line`, `dates_text` |
| `tools/weather.py` | `find_places` (top matches, one per name + state + country), `fetch_hours`, `outing_hours`, the rain-chance fill |
| `tools/climate.py` | `archive_hours`: last year's hours for a range of days |
| `api/packing.py` | `POST /packing/preview`, `POST /packing`; the city defaults to the home city |
| `ui/packing_page.py` | trip form, events, the confirmation card, the plan |
| `scripts/pack_trip.py` | the same in the terminal: prints the check, `--place N`, asks before packing |
| `tools/variety.py`, `scripts/check_variety.py` | the repeat check above |
| `cli.py` | `trip_event`: `--event 2,party,20:00,4,0.5,car` |

Tests: `test_packing.py` (slots, packing, limits, weather with a fake Open-Meteo,
preview and confirmation), `test_api_packing.py`, `test_ui_packing_page.py` (AppTest
against the API over fakes), `test_cli.py`.

## Live check (2026-10-25, 26-item wardrobe, no DeepSeek calls)

The page, driven headless against the real API: Manali, 4 days, parties on days 2 and
4.

- The check offered Manali, Tamil Nadu first (34-39 °C) and Himachal Pradesh second;
  picking Himachal re-checked: 10-18 °C by day, 3-7 °C at the parties.
- Packed 7 pieces: two tops (light pink button-down shirt, red polo), two bottoms
  (black trousers, grey chinos), tan hiking shoes, a blue webbing belt for the days
  and a black belt for the parties. No top more than 2 days, no bottom more than 3.
- Gaps, all real: no warm outerwear (days 1-4), no scarf and no party shoes for the
  cold evenings.
- Not checked: a trip with dresses or ethnic wear (the wardrobe has few), and how the
  page looks in a real browser (AppTest checks its content, not its layout).
