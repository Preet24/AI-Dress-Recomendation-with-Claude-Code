# User guide

How to install, start and use the outfit recommender. (How it is built is in the
[README](../README.md) and [`phases/`](phases/README.md).)

## What you need

- Python 3.11 or newer, with PyTorch already installed (the CPU build is enough).
- A DeepSeek API key, for tagging photos and for the chat.
- No GPU: everything that runs on your machine runs on the CPU.

## Install (once)

```powershell
cd <the project folder>
.\.venv\Scripts\Activate.ps1              # your virtual environment
pip install -e ".[api,ui,dev]"
copy .env.example .env                    # then open .env and set DEEPSEEK_API_KEY
```

The first photo you add downloads the cutout and CLIP model weights (a few hundred MB).

## Start the app

```powershell
python scripts/run_app.py
```

This starts the server and the web app, and opens http://localhost:8501 in your
browser (ready in about 20 seconds). Keep the terminal open while you use it;
**Ctrl+C** in the terminal stops everything.

| Option | Use it to |
|---|---|
| `--no-browser` | not open the browser (open http://localhost:8501 yourself) |
| `--port 8080`, `--ui-port 8502` | use other ports, if 8000 / 8501 are taken |
| `--record` | save every chat turn as JSON in `data/conversations/` |

The left sidebar has the six pages, **Chat**, **Wardrobe**, **Shopping**, **Packing**,
**History** and **Profile**, and shows how many garments you have.

## Wardrobe page: your clothes

**See what you have.** Every garment is shown as a cut-out picture with its tags
(category, formality and warmth from 1 to 5, occasions). Narrow the grid with the
**Categories** and **Occasions** filters; dresses and jumpsuits are under "one piece".

**Add clothes.** Open **Add a photo**, choose a JPEG, PNG or WEBP file (up to 15 MB),
and say what the photo shows:

- **Garment alone**: laid flat or on a hanger; gives one garment.
- **Worn by a person**: gives the top and the bottom as two garments, or one for a dress.

Then click **Add to wardrobe**. The app cuts the garment out, tags it, measures its
color and indexes it. This takes about half a minute per photo on a CPU (the first one
a little longer while the models load). Uploading the same photo again updates its
garments instead of adding copies.

Tips for clean cut-outs:

- one garment per flat photo, filling most of the frame, with nothing else touching it;
- a plain background in a different color from the garment (a beige item on a beige
  sheet is hard to cut out);
- a hanger is fine: its hook is trimmed, but the part across the collar stays in the
  picture, so take it off for the cleanest result;
- good, even light.

**Accessories** (belts, ties, bags, sunglasses, hats, scarves, dupattas, watches,
jewellery): photograph each one alone and choose **Garment alone**. A belt or
sunglasses in a worn photo are not cut out separately. Check that the tags say
category "accessory" and a subtype that names it ("brown leather belt", "silk tie");
fix it with **Edit** if not. Socks count as accessories too (outfits leave them out);
if a pair was tagged "bottom" or "footwear", set it to "accessory".

**Fix a garment.** Open **Edit** on its card:

- Change any tag in the form and click **Save changes** (only what you changed is saved).
- Or type what it really is ("it's a midi skirt, not shorts") and click **Re-tag**:
  the tagger redoes all its tags using your note.

**Remove a garment.** **Delete** on its card, then **Yes, delete**. This removes the
garment and its cut-out picture and can't be undone (your original photo stays in
`data/wardrobe_raw/`).

## Chat page: get an outfit

Say the occasion, and if you like where and when:

> party in Pune tonight at 8

If something is missing, the bot asks (how long you'll be out, how much of it outdoors,
how you'll travel) and then recommends. Each recommendation shows:

- **Top pick** and **Another option** cards with pictures of your own garments, why
  they go together, and any warnings ("the jeans are a bit casual for a party");
- the accessories that finish each outfit, if you own any: at most one belt, tie,
  bag, watch ... each. A belt only with trousers or jeans (matching the shoes), a tie
  only with a shirt for a dressy occasion, sunglasses only in strong sun or on a ride,
  a scarf only in the cold. They never change which outfit is picked;
- what your wardrobe is missing for this occasion, if anything;
- **How this turn ran**: which agents ran, how many DeepSeek calls, how long it took.

Then change it in your own words:

| You say | It does |
|---|---|
| "something with my blue jeans instead" | builds the outfit around that garment |
| "swap the shoes", "less formal" | changes the suggestion |
| "what if I go by scooter?", "it might rain" | re-checks the same outing with new details |
| "show me my other tops" | shows matching garments as pictures |
| "what about office tomorrow at 9?" | starts a new request |

It only ever suggests clothes you own; anything you'd need but don't have is named as
missing ("a ..."), not invented. The weather comes from Open-Meteo for the city and
time you give.

**New conversation** in the sidebar starts over. Reloading the page keeps the
conversation, but stopping the app forgets all conversations (your wardrobe is kept).

## Shopping page: what to buy, and should I buy this?

First tick the **occasions you dress for** and press **Save occasions**. Your saved
places (Profile page) always count too; with nothing ticked and no places, both tabs use
casual, office and party.

**What should I buy?** (free, no DeepSeek call)

1. Choose the occasions to **Shop for** (it starts with yours) and press **Suggest**.
   It takes a few seconds.
2. **What your wardrobe is missing** lists occasions with no good outfit (and the
   pieces that came closest, with why they don't fit), and gaps such as "no footwear
   suits office", with the weather they happen in.
3. Then up to five suggestions, best first, such as **black loafers** or **navy
   trousers**: why (which occasions it makes possible, which gaps it fills, how many new
   outfits), why that color ("navy goes with 9 of your 9 ..."), and pictures of your
   pieces it **goes with**. Some come in pairs ("buy it together with the navy track
   pants"); later ones count the earlier ones as bought, so they never overlap.

Suggestions come from a list of about 30 common pieces, skip colors you avoid (Profile
page) and colors you already own that piece in, and use your gender if you set it.
Sunglasses, a wool scarf or a tie are suggested only to fill a missing accessory
("no sunglasses for the strong sun", "no scarf for the cold"), and after clothes that
fill a gap.

**Should I buy this?**

1. Add a photo of the item (a shop photo or screenshot
   works), say whether it is worn by a model or shown alone, and press **Check it**.
   It is cut out and tagged like a wardrobe photo (one DeepSeek call per garment found)
   but **not added**.
2. Each garment gets a card: **Buy**, **Maybe** or **Skip**, with the reasons (how many
   new outfits it gives, which occasions and weather it makes possible, gaps it fills
   such as "no footwear suits party", how its color goes with your pieces), pictures of
   what it **goes with** and **clashes with**, and the **closest piece you own**.
3. Bought it? **Add to wardrobe** stores it as tagged, with no second DeepSeek call.

An accessory (photographed alone) gets a card too: how many of your outfits it would
finish, **Buy** only if it fills a missing accessory, **Maybe** if it just goes with
your outfits, **Skip** if none of them calls for it.

The weather it plans for is your home city's (Profile page): last year's daily weather
decides which of hot, mild, cold and rainy days you get. Without a home city it plans
for all four. Checked items wait until the app stops; the 20 newest are kept.

## Packing page: what to take on a trip

1. **Describe the trip:** a **town or city** (not a state: "Goa" finds Genoa in
   Italy, so type Panjim or Calangute; empty means your home city), the first day and
   how many days (up to 14), what you usually do each day (occasion, leaving time,
   hours away and outdoors, transport), whether the first and last day are travel
   days, and how many days you'll wear one top, bottom or dress (2, 3 and 1 to start
   with; set dresses to **0** to pack none). Tick **Evenings in (relax / sleep)** for
   something to lounge and sleep in every night: no shoes, no weather, and only pieces
   tagged **lounge** count (tag your joggers, shorts or pyjamas lounge on the Wardrobe
   page).
2. **Add events** such as a dinner or a party under **Add an event** (day, occasion,
   time, hours, transport); **Remove** takes one off.
3. Press **Check trip**. Nothing is packed yet: the **Is this right?** card shows
   - **Which place?** when the name fits several: each with its state and country
     (Manali, Tamil Nadu or Manali, Himachal Pradesh); pick the right one and the
     weather is checked again for it;
   - **Where** and **When** ("Fri 30 Oct - Mon 2 Nov 2026, 4 days");
   - a row for every day and event: time, occasion, hours out, transport, the weather
     (feels-like, rain chance, wind, UV, sky) and where it came from: the forecast, or
     **last year on the same dates** when the trip is more than about two weeks away.
4. If it's right, press **Pack for this**. You get the pieces to pack with the days
   each is worn, then a card per day and event with its outfit, weather and any
   warnings. It doesn't put the same top and bottom together on two days, even if that
   means packing one more piece; a warning ("your navy shirt again: ... nothing else
   fits") tells you when your wardrobe leaves no other choice. **Missing from your wardrobe** lists what would help ("no scarf for the
   cold (days 2 and 4)"). When nothing fits an event, it says which piece is short
   and why ("no bottom fits: grey chinos is not tagged for formal and has formality
   3, wanted 4-5"). For formal events and weddings, plain trousers or a skirt one point
   below may still go with a dressy enough top.

Changing the trip or the events, or pressing **Check trip** again, clears the old check
and plan. Packing uses your profile's colors and fits, as the chat does. It is free: no
DeepSeek call, only the weather.

## I wore this, and the History page

Under every outfit in **Chat** and every day in **Packing**, open **I wore this**:

1. Untick any piece you didn't wear, pick how it was (**Loved it**, **Fine**, **Not
   again**: required), and what was off if anything (too warm, too cold, too formal,
   too casual, colours didn't go, uncomfortable). The date and occasion are filled in
   (a packing day uses its own); a note is just for you.
2. Optionally add a **photo of you wearing it**. It stays on this computer (a smaller
   copy without location or camera data) and is compared with your wardrobe locally.
3. Press **Log it**. It's free.

What the log changes, a little at a time and never as a ban:

- a top + bottom (or a dress + shoes) you **loved** ranks higher next time; one marked
  **not again** or "colours didn't go" ranks lower;
- what you wore in the last 7 days ranks a little lower, so suggestions vary (chat
  only; packing ignores it);
- after you've found outfits **too cold** two more times than too warm (or the other
  way round), suggestions pick one warmth point warmer (or lighter), and the reply
  says why.

The **History** page lists what you logged, newest first. For an entry with a photo:

- **What the photo shows:** ✅ "your tan chinos: yes"; ❓ "looks more like your grey
  chinos than your navy trousers: which was it?" with a **Use your grey chinos**
  button; ➕ a piece you didn't log (**Add your ...**); ⚠️ something that may not be
  in your wardrobe. Shoes and accessories can't be checked from a photo and are listed
  as not checked. The check is a guess: dark trousers often look alike, so it only
  asks when another piece is clearly closer, and you decide.
- **Ask DeepSeek about this look** sends the photo (shrunk, without location data) to
  DeepSeek once. You get scores from 1 to 5 with a reason each for fit, proportion,
  colour on you (using your colouring, if you analysed it), coordination, right for the
  occasion and polish, plus one or two tips. It scores the outfit, never you: nothing
  about your body, face or looks. The scores are kept with the entry and change
  nothing else; a new photo clears them.
- **Remove photo**, or add one later; **Delete** removes the entry and its photo.

## Profile page: about you and your places

Everything here is optional.

- **About you:** gender, home city, height, build, waist, top size, the fit you like
  for tops and for bottoms, and colors you like or avoid (comma-separated). Press
  **Save**.
- **Your places:** places you go often, such as Office, Gym or College. Each has an
  occasion, a dress code (formal 4–5, business 4, business casual 3–4, smart casual 3,
  casual 2–3; leave it unset for the occasion's usual range), and your usual trip:
  start time, hours away, hours outdoors and transport. Open a place to change or
  remove it; **Add a place** adds one. Names must be unique.

The profile is saved on this computer only. The chat uses it:

- name a saved place ("office tomorrow") and it takes the occasion and dress code
  from it, then asks "Same as usual for Office: leaving at 09:00, 9 h away, ...?".
  Say **yes**, give only what's different ("yes, but by scooter"), or **no** to answer
  the usual questions instead;
- leave out the city and it uses your home city, and says so in the reply;
- outfits in colors you like and in your preferred fits rank a little higher; colors
  you avoid rank lower but still appear when nothing else fits (only a garment's main
  color counts).

Only the facts a reply needs (your saved place names, and the request built from
them) are sent to DeepSeek.

**Photo and colouring (optional).** Upload a daylight photo of your face and press
**Save photo**: it is kept only on this computer, as a smaller copy without location or
camera data. **Analyse my colouring** sends that copy (shrunk again) to DeepSeek once
and fills in your colouring ("warm undertone, medium depth, ..."), colors that suit
you and colors to go easy on. It judges only undertone, hair and eye color: never age,
ethnicity or body shape. Edit the text if it is off; empty the first box to clear it.
You can delete the photo afterwards and keep the text. In ranking, colors that suit
you count a little, colors to go easy on count a little against, and your own likes
and avoids always win.

## What costs money

Only DeepSeek calls: a few per chat message, one per garment when adding a photo, one
per colouring analysis, one per garment in a Shopping check,
one per re-tag, one per **Ask DeepSeek about this look**. Browsing, filtering, editing
tags in the form, deleting, "What should I buy?", packing, **I wore this** and the photo
check are free.
Weather is free.

## What leaves your machine

Your photos and the wardrobe index stay local. Sent out: each garment's cut-out (to
DeepSeek, when added, re-tagged or checked on the Shopping page), the outfit facts for a reply (to DeepSeek, when
chatting), the city name (to Open-Meteo), your profile photo (to DeepSeek, only
when you press **Analyse my colouring**), and a worn photo (to DeepSeek, only when you
press **Ask DeepSeek about this look**).

## Where things are kept

| Folder | What |
|---|---|
| `data/wardrobe_raw/` | your original photos |
| `data/wardrobe_clean/` | the cut-out pictures |
| `qdrant_data/` | the wardrobe index (tags and image vectors) |
| `data/conversations/` | saved chat turns, only with `--record` |
| `data/profile.json` | your profile and saved places |
| `data/profile_photo.jpg` | your profile photo, if you saved one |
| `data/candidates/` | photos and cut-outs of items you checked but haven't added |
| `data/wear_log.json` | outfits you logged with **I wore this** |
| `data/worn/` | photos of you wearing them (one per entry) |

## Other ways to use it

```powershell
python scripts/ingest_folder.py data/wardrobe_raw --mode flat   # add a whole folder of photos
python scripts/review_wardrobe.py                               # check tags one by one in the terminal
python scripts/chat.py                                          # chat in the terminal instead of the browser
python scripts/pack_trip.py --city Manali --first-day 2026-11-06 --days 4 --place 2   # pack in the terminal (asks first)
python scripts/serve.py                                         # only the server (API docs at http://127.0.0.1:8000/docs)
```

Only one of these can use the wardrobe at a time: stop the app before running them.

## Troubleshooting

| You see | Do this |
|---|---|
| `Port 8000 is already in use` | The app (or `serve.py` / `streamlit`) is already running: use that, stop it, or pass `--port` / `--ui-port` |
| `The API stopped while starting` | Read the messages above it. Most often `chat.py` or `review_wardrobe.py` is still open (only one program can open the wardrobe), or `.env` has no `DEEPSEEK_API_KEY` |
| "Can't reach the API ..." in the browser | The server isn't running or is still starting: start it, wait, refresh |
| "Couldn't reopen that conversation" | The app was restarted, so old conversations are gone; just start a new one |
| A photo is refused | Use JPEG, PNG or WEBP under 15 MB; a photo in which no garment is found is refused too |
| A garment is tagged wrong | Wardrobe → **Edit** → fix the field, or **Re-tag** with a note |
| Packing finds the wrong place, or "No place found" | Type a town or city, not a state or region (Panjim, not Goa), and check its state and country in **Which place?** |
| History says a piece "may not be in your wardrobe" | The photo check found nothing close: use a well-lit photo where you fill more of the frame, or add that garment on the Wardrobe page |
| "No one is clearly wearing the outfit" | Ask DeepSeek about this look needs a photo of you wearing it, head to toe if possible |
