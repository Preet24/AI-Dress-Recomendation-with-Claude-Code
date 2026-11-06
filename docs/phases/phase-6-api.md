# Phase 6 — API

Phases 2–5 were Python functions and terminal scripts, each one opening the wardrobe,
loading CLIP and building the agent graph itself. Phase 6 puts all of it behind one
FastAPI server, so the Phase 7 UI (or any other client) only sends HTTP requests. It
adds no new logic: every endpoint wraps code that already existed and was tested.

```
python scripts/serve.py              # http://127.0.0.1:8000, interactive docs at /docs
python scripts/serve.py --record     # also save each chat turn to data/conversations/
```

## Endpoints

| Endpoint | Wraps | Returns |
|---|---|---|
| `GET /health` | `WardrobeStore.count` | status, item count, agent model |
| `GET /items?category=&occasion=` | `WardrobeStore.list_items` | the wardrobe, tops first; repeat a filter to allow several |
| `GET /items/{id}` | `WardrobeStore.get` | one item |
| `GET /items/{id}/image` | the cutout file | the background-removed PNG |
| `POST /items` (photo + `mode`) | `Pipeline.ingest_image` | the garments found (201); re-uploading a photo updates them |
| `POST /items/{id}/retag {note}` | `review.retag_with_note` | the item re-tagged with your correction |
| `PATCH /items/{id}` | `review.apply_field_edits` | the item with the tags you sent changed |
| `DELETE /items/{id}` | `review.remove_item` | 204; the item and its cutout are gone |
| `POST /chat {message, thread_id?}` | `recording.run_turn` | a reply or a question, plus this turn's outfits, gaps, shown items, path, calls, time |
| `GET /chat/{thread_id}` | the graph's saved state | the messages so far and the open question, if any |

Items are returned with `name` (what the agents call it, "lavender skirt") and
`image_url`, never the local `image_path`.

### Added after Phase 7: profile and shopping

The user profile and the shopping assistant (see `../features_to_add.md` and
[`shopping-assistant.md`](shopping-assistant.md)) added two routers in the same style:
plain `def` endpoints over `Services`, errors mapped in `api/errors.py`.

| Endpoint | Wraps | Returns |
|---|---|---|
| `GET /profile`, `PUT /profile` | `ProfileStore.load` / `save` (`data/profile.json`, written atomically) | the whole profile; every save sends all of it |
| `PUT /profile/photo`, `GET /profile/photo`, `DELETE /profile/photo` | `ProfileStore.save_photo` (`reencode`: upright JPEG, ≤ 1024 px, no EXIF) | 204 / the kept JPEG / 204 (the colouring text stays) |
| `POST /profile/analyse` | `colouring.analyse_colouring` (1 DeepSeek call) | the profile with its colouring filled in; 422 if no face is visible |
| `POST /shopping/suggest {occasions}` | `staples.missing` + `staples.suggest_purchases` (no DeepSeek call) | what the wardrobe is missing, staples to buy with reasons and the user's pieces they go with |
| `POST /shopping/check` (photo + `mode`, default worn) | `Pipeline.describe_photo` + `shopping.check_candidate` | one report per garment found (buy / maybe / skip and why); nothing stored |
| `GET /shopping/candidates/{id}/image` | the candidate's cutout | PNG |
| `POST /shopping/candidates/{id}/add` | `CandidateStore.add_to_wardrobe` | the stored item (201), with no second tagging call |

Candidates wait in memory (`CandidateStore`, newest 20) and under
`data/candidates/`. The home city's typical weather comes from `Services.climate`,
cached per city and day. `fake_services` gives tests an in-memory profile, so they
never touch `data/`.

A chat, as the UI sees it:

```
POST /chat {"message": "what should I wear to a party in Pune tonight at 7?"}
→ {"thread_id": "a280…", "kind": "question", "path": ["router", "intake"],
   "text": "Got it. About how many hours will you be out? ..."}

POST /chat {"message": "about 4 hours, half an hour outside, going by cab", "thread_id": "a280…"}
→ {"kind": "reply", "path": ["ask", "intake", "orchestrate", "stylist", "critic", "respond"],
   "text": "For the party in Pune tonight, go with your black button-up shirt, ...",
   "outfits": [{"pieces": [{"name": "black button-up shirt", "image_url": "/items/0fdf…/image", ...}, ...],
                "reasons": ["colors: neutrals with a lavender accent"], "warnings": []}, ...],
   "gaps": [], "llm_calls": 3, "seconds": 7.07}
```

## Design decisions

**One process owns the wardrobe.** Local Qdrant allows only one process to open
`qdrant_data/`. At startup `live_services` opens it once and hands that store to both
ingestion (`Pipeline`) and the agents (`clip_context`); both already took the store
as an argument, so no existing code changed. CLIP and rembg load on the first photo
upload, not at startup.

**Dependencies come in through `Services`.** `create_app(services)` takes everything
it needs; tests pass an in-memory store, the fake DeepSeek client and word-overlap
search (`tests.fakes.fake_services`), so the API tests use no network, no CLIP and no
real wardrobe.

**Errors are mapped once.** Domain errors become status codes in `api/errors.py` only:
unknown item or conversation 404, busy conversation 409, bad edit or unusable photo
422 (also no face in a profile photo), DeepSeek failure 502 (logged).

**Endpoints are plain `def`.** The store, CLIP and the agent graph are blocking calls;
FastAPI runs sync endpoints in its thread pool, so one slow chat doesn't stop the rest.

### Uploads trust the bytes, not the client

- The filename and content type the client sends are ignored. The image is decoded
  completely; only JPEG, PNG and WEBP are accepted, up to `max_upload_mb` (15).
  `verify()` alone was not enough: a test showed it passes a truncated JPEG, which
  then crashed the pipeline with a 500 instead of a 422.
- The saved name keeps only a cleaned stem inside `raw_dir` (`../../etc/passwd` →
  `passwd.jpg`); a different photo with the same name gets a content hash added.
- If tagging fails or no garment is found, nothing is stored and the photo just saved
  is removed. Uploads and re-tags run one at a time (`ingest_lock`).

### Chat: the server keeps the conversation

- **The client only sends messages.** Whether a message is a new request or the answer
  to a question comes from the graph's saved state (`get_state(config).next`: paused at
  the `ask` interrupt), never from the client. An answer resumes at `ask`, so it makes
  only the calls still needed.
- **Exact records under concurrency.** `RecordingClient` logs calls in one list, so
  two chats sharing it would mix their calls. Each turn instead compiles the graph
  around its own `RecordingClient` (milliseconds); conversations stay shared through
  one checkpointer.
- **One message at a time per conversation.** A second message while the first is
  still being answered gets 409 and never reaches the model.
- **What the UI needs, not just text.** `run_turn` now also returns the turn's plan and
  the items it showed: by the end of the turn the router has cleared the old plan and
  respond has reset `found`, so the final state no longer has them.
- **The open question survives a page reload.** The ask node adds its question to
  `messages` only when the answer resumes the run, so `GET /chat/{id}` reads it from
  the saved interrupt instead. A test assumption caught this gap.

## Limits

| Limit | Why it is acceptable here |
|---|---|
| Conversations are kept in memory and lost on restart (an old `thread_id` gets 404) | A SQLite checkpointer needs new packages and upgrades `langgraph-checkpoint` 4.2 → 4.3; not worth the risk for a single-user demo |
| No login; serves on `127.0.0.1` by default (`serve.py` warns for any other host) | It is your own wardrobe on your own machine |
| One worker process | Local Qdrant allows one process; a Qdrant server would lift this |

## Live check

Against the real wardrobe over real HTTP (`serve.py`, then an `httpx` client), with 5
DeepSeek calls and nothing written: `/health` reported 10 items; `/items` listed them
tops first; a cutout came back as a 279 KB PNG; an unknown id gave 404. The chat above
asked one combined question (2 calls, 10.8 s), the answer resumed at `ask` and gave a
recommendation with three outfits (3 calls, 7.1 s), and `GET /chat/{id}` showed the
open question while waiting and the four messages afterwards. The reply carried the
Phase 5 fixes: "about half an hour in total", an alternative introduced as "if you'd
rather wear a dress".

## Files

| File | Job |
|---|---|
| `api/services.py` | `Services`, `live_services`, `get_services`; per-turn `graph(client)`, thread locks |
| `api/errors.py` | `NotFound`, `BadUpload`, `Busy` and the status-code table |
| `api/items.py` | wardrobe and upload endpoints, `ItemOut`, `TagEdits` (built from `TagResult`) |
| `api/chat.py` | chat endpoints, `ChatOut`, `Conversation` |
| `api/profile.py` | profile, profile photo and colouring analysis (added later) |
| `api/shopping.py` | what to buy, check a candidate, candidate image, add (added later) |
| `api/app.py` | `create_app(services=None, settings=None)`, `/health` |
| `scripts/serve.py` | uvicorn on 127.0.0.1, `--port`, `--record` |

## Testing

FastAPI's `TestClient` drives every endpoint over fake services: filters and ordering,
no local path in responses or in the OpenAPI schema, seven kinds of bad edit (each 422,
item unchanged), worn-photo upload storing two garments, re-upload updating them, bad
files rejected before anything runs, failed tagging leaving nothing behind, unsafe
filenames, re-tagging with a note, a question then an answer (counting model calls to
prove nothing repeats), reading a conversation back, items shown in a follow-up,
unknown and busy conversations, and recording. Files: `test_api.py`,
`test_api_items.py`, `test_api_upload.py`, `test_api_chat.py`; later
`test_profile.py`, `test_profile_photo.py` (profile endpoints, photo re-encoding,
analysis with a fake reply) and `test_api_shopping.py` (a check storing nothing,
add without re-tagging, suggestions making no DeepSeek call).
