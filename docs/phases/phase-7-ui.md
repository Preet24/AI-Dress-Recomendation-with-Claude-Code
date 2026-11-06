# Phase 7 — UI

Phase 6 put the wardrobe and the agents behind an HTTP API. Phase 7 adds a Streamlit
app on top of it: chat with the agents and see the recommended outfits as pictures of
your own clothes, and browse, add, fix and remove garments without the terminal. Like
the API, it adds no outfit logic: every page shows what the API returns.

```
python scripts/run_app.py                # both, in order; opens the browser; Ctrl+C stops both

python scripts/serve.py                  # or by hand: the API, http://127.0.0.1:8000
streamlit run src/outfit/ui/app.py       # the UI, http://localhost:8501
                                         # set OUTFIT_API_URL if the API is elsewhere
```

`run_app.py` (logic in `outfit/launcher.py`) checks both ports are free, starts
`serve.py`, waits for `/health` (it reports the exit code if the API dies first, e.g.
when `chat.py` still holds the wardrobe), then starts Streamlit headless with
`OUTFIT_API_URL` set, waits for it, and opens the browser. When either process ends,
or on Ctrl+C, it stops the other. Real start on this machine: about 18 s.

## Pages

| Page | What you can do | API calls |
|---|---|---|
| **Chat** (default) | Ask for an outfit, answer the bot's questions, follow up ("something with my jeans instead"). Each recommendation shows a card per outfit (top pick, then alternatives) with the cutouts, the reasons and any warnings, plus the gaps in your wardrobe and a "How this turn ran" trace (agent path, DeepSeek calls, seconds) | `POST /chat`, `GET /chat/{id}`, cutout images |
| **Wardrobe** | Grid of every garment with its tags, filter by category and occasion, add a photo (flat-lay or worn), edit tags in a form, re-tag with a note ("it's a midi skirt"), delete with a confirmation | `GET/POST /items`, `PATCH`, `POST …/retag`, `DELETE` |
| **Shopping** (added later) | Tick the occasions you dress for. **What should I buy?**: choose occasions, get what the wardrobe is missing and up to five pieces to buy, with why and pictures of what each goes with. **Should I buy this?**: upload a shop photo, get buy / maybe / skip per garment, what it goes with and clashes with, the closest piece you own, and **Add to wardrobe** | `GET/PUT /profile`, `POST /shopping/suggest`, `POST /shopping/check`, candidate images, `…/add` |
| **Profile** (added later) | About you (gender, home city, sizes, fits, liked and avoided colors), saved places with their dress code and usual trip, an optional face photo and colouring analysis | `GET/PUT /profile`, `/profile/photo`, `POST /profile/analyse` |

Since Phase 7 the profile and shopping assistant added the last two pages; they follow
the same rules below (API only, tested headless).

The sidebar shows the wardrobe size and agent model from `/health`; if the API is
down the app stops with how to start it, instead of failing page by page.

## Design decisions

**The UI only talks to the API.** Every request goes through `ui/client.OutfitAPI`
(one method per endpoint, typed responses, errors as `APIError(status, detail)` with
the server's message). The UI never imports the server side: no torch, Qdrant, openai,
FastAPI, LangGraph, agents or ingestion code (only `outfit.schemas` for its enums). A
test starts a fresh interpreter, imports the UI and checks the loaded modules, so this
can't break silently. So the UI loads no models, and it could run on another machine
or in another container from the API.

**The conversation lives on the server, its id in the URL.** After the first message
the page sets `?thread=<id>`. A reload (or a bookmarked link) reads the conversation
back with `GET /chat/{id}`, including a question still waiting for an answer. An
unknown id (the server restarted) gives a warning and a fresh chat, not an error page.
Restored replies show text only: outfits belong to the turn, not the saved messages.

**Edits send only what changed.** The edit form holds every tag, but `tag_changes`
compares it with the item and sends only the changed fields, so saving an unchanged
form calls nothing ("Nothing changed"), and the success line names the fields
("Saved lavender skirt: length, formality"). Validation stays in the API (`TagEdits`,
built from the tagger's own schema), and its message is shown in the form.

**Destructive actions ask first.** Delete shows "Delete … and its cutout? This can't be
undone." with Yes / Cancel; after any change the page reruns and shows a one-line
result at the top.

**Pictures are cached.** Cutouts are fetched once through the API and kept in
`st.cache_data` (300 images, 10 minutes), so moving between pages and filtering
doesn't refetch them. A missing picture shows "(no picture)" rather than breaking the
grid.

## Live check

The real `app.py` against `serve.py` on the real 10-item wardrobe, with 5 DeepSeek calls
and nothing written:

- The sidebar showed "10 items in your wardrobe · deepseek-flash".
- **Chat.** "party in Pune tonight at 8" got one combined question about hours out,
  time outdoors and transport (router → intake, 2 calls, 7.4 s). The answer "by cab,
  about 15 minutes outside, back by midnight" resumed at `ask` and returned three outfit
  cards with 8 cutouts (ask → intake → orchestrate → stylist → critic → respond, 3 calls,
  9.4 s):

  > For the party in Pune tonight, go with your black button-up shirt, lavender skirt and
  > champagne slingback heel pumps. The black lets the lavender stand out. You're only
  > outside about 15 minutes in total by cab, and it's clear and warm (27-28°C), so this
  > suits the evening well. If you'd rather wear a dress, your blush pink floral midi
  > dress with the same pumps is a lovely switch. …

- **Reload.** A new session opened with the `?thread=` id restored all five bubbles,
  with the same final reply.
- **Wardrobe.** All 10 garments rendered with their pictures, tops first; the
  "bottom" filter left the beige trousers, lavender skirt and blue jeans.
- `streamlit run` served the app (`/_stcore/health` ok). The API log showed only 200s.

Upload, edit, re-tag and delete were not run on the real wardrobe (they would change
it or call DeepSeek); the tests below cover them against fakes.

## Files

| File | Job |
|---|---|
| `ui/client.py` | `OutfitAPI`: one method per endpoint → `Item`, `ChatTurn`, `Conversation`, `Health`, `Profile`, `ShoppingCheck`, `ShoppingSuggestions`; `APIError` |
| `ui/widgets.py` | cached cutout pictures, item grid, labels, one-line flash messages (shared by the pages) |
| `ui/shopping_page.py` | occasions form; "What should I buy?" and "Should I buy this?" tabs (added later) |
| `ui/profile_page.py` | about you, saved places, photo and colouring (added later) |
| `ui/chat_page.py` | chat bubbles, outfit cards, gaps, trace; `?thread=` restore; New conversation |
| `ui/wardrobe_page.py` | upload, filters, item cards with edit / re-tag / delete |
| `ui/app.py` | entry point: API client (`OUTFIT_API_URL`), health in the sidebar, navigation |
| `launcher.py`, `scripts/run_app.py` | one command: API, then UI, browser; stops both together |

## Testing

Streamlit's `AppTest` runs the pages headless against the real FastAPI app over fake
services (`OutfitAPI(http=TestClient(create_app(fake_services(...))))`), so each UI test
goes through the client, the HTTP layer and the API code, with fake DeepSeek replies and
no network:

- **Client** (`test_ui_client.py`): every model parses real API responses; server
  errors and an unreachable API become readable messages; the import-isolation check.
- **Chat** (`test_ui_chat_page.py`): a question then a recommendation with its pictures
  and trace; a reload restoring an open question; an unknown thread; New conversation;
  a failed turn shown in the chat; the full `app.py` opening on Chat with the wardrobe
  size, and stopping with instructions when the API is down.
- **Wardrobe** (`test_ui_wardrobe_page.py`): grid order and captions, filters, an edit
  saving only the changed fields, an unchanged form, an edit the API refuses, re-tag with
  a note (the note reaches the tagger), delete with cancel then confirm, the empty
  wardrobe.
- **Shopping** (`test_ui_shopping_page.py`): report cards from a check, saving the
  ticked occasions, Add to wardrobe once, Suggest showing the missing list and a card
  per suggestion.
- **Profile** (`test_ui_profile_page.py`): the about-you form, adding, changing and
  removing places, and that the client's mirrored enums match the server's.

The launcher (`test_launcher.py`) runs with fake processes: start order, the UI pointed
at the API, a busy port refused before anything starts, an API that dies while
starting, a timeout, terminate-then-kill, and Ctrl+C giving both time to stop.

`AppTest` can't drive `st.file_uploader`, so uploads are tested one level down: the
client against the API's upload endpoint, which `test_api_upload.py` covers with fake
rembg and CLIP.

## Not done: Docker

The plan included a CPU-only container. It was left out for now: the image would be
about 1.5 GB to download (CPU torch, ONNX Runtime, CLIP and rembg weights) for a demo
that runs from the venv. The split above already fits it: the API would be one
container owning `qdrant_data/`, and the UI could be a second, small one pointed at it
with `OUTFIT_API_URL`.
