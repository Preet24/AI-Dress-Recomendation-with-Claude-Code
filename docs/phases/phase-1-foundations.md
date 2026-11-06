# Phase 1 — Foundations

The groundwork every later phase builds on: one data model for a garment, one place
for settings, one storage layer, and guardrails for the development environment.

## What it delivers

| Piece | File | Job |
|---|---|---|
| Data model | `src/outfit/schemas.py` | `ClothingItem` and its enums; deterministic item ids |
| Settings | `src/outfit/config.py` | Every tunable value, overridable from the environment |
| Vector store | `src/outfit/db/qdrant_store.py` | `WardrobeStore`: one Qdrant point per garment, filtered vector search |
| Packaging | `pyproject.toml` | Dependencies (without torch), pytest and ruff config |
| SSL fix | `src/outfit/_ssl_fix.py` | Keeps the shared venv's `pip-system-certs` from breaking HTTPS |
| Secret guard | `.claude/hooks/block-env.ps1`, `.claude/settings.json`, `.ignore` | Claude Code can never read `.env` |

## The data model

`ClothingItem` is the single source of truth for a garment; every other part of the
system reads or writes it.

| Field | Type | Notes |
|---|---|---|
| `id` | UUID | Deterministic, see below |
| `image_path` | Path | The background-removed PNG cutout |
| `category` | enum | top, bottom, one_piece, outerwear, footwear, ethnic_top, ethnic_bottom, accessory |
| `subtype` | str | Free text: "kurta", "jeans", "blazer" |
| `primary_color_name` | str | Named by the vision model |
| `primary_color_lab` | (L, a, b) | **Measured locally**, never taken from the model |
| `secondary_colors` | list[str] | |
| `pattern` | enum | solid, striped, checked, floral, printed, other |
| `fabric` | str | Best guess: cotton, denim, linen, wool, silk, synthetic… |
| `length` | enum | cropped, short, knee, midi, full |
| `fit` | enum | slim, regular, relaxed, oversized |
| `formality` | 1–5 | 1 = gym, 5 = black tie / wedding |
| `warmth` | 1–5 | 1 = very light, 5 = heavy winter |
| `flowy` | bool | Matters in wind and when riding |
| `occasions` | list[enum] | casual, office, formal, sports, party, indian_festive, wedding, travel, lounge |
| `created_at` | datetime | Kept when a photo is re-ingested |

Design choices:

- **Enums wherever a rule depends on the value.** Category, occasion, pattern, length
  and fit are closed sets, so the outfit rules in Phase 3 can match on them reliably.
  Values only a human reads (subtype, fabric, color name) stay free text.
- **`extra="forbid"`**: a typo'd or invented field is a validation error, not silently
  stored data.
- **Stable ids.** `stable_item_id(photo_bytes, region)` is a `uuid5` of the photo's
  SHA-256 plus the garment region ("upper", "lower"…). Re-ingesting the same photo
  produces the same ids, so it updates items instead of duplicating them.

## Settings

`Settings` (pydantic-settings) reads `.env` and the environment. Every field can be
overridden with an `OUTFIT_<FIELD>` variable, so experiments (another CLIP model, a
different segmentation model) need no code change. Groups:

- segmentation: `flat_model`, `worn_model`, `max_side`, `min_area_ratio`;
- embeddings: `clip_model`, `clip_pretrained`, `embed_batch_size`;
- DeepSeek: `deepseek_api_key` (a `SecretStr`, read from `DEEPSEEK_API_KEY`),
  `tagger_base_url`, `tagger_model`, `agent_model` and their token limits;
- storage: `clean_dir`, `qdrant_path`; added later: `raw_dir`, `conversations_dir`,
  `max_upload_mb` and `record_chats` (API), `profile_path` (the user profile) and
  `candidates_dir` (shopping items checked but not bought).

The key is a `SecretStr`, so it never shows up in a log line or a `repr`.

## The vector store

Qdrant runs in **local mode** (`QdrantClient(path="qdrant_data")`): no server, the
index is a folder on disk. Collection `wardrobe`:

- one **named vector** `image` (512 dimensions, cosine): the CLIP embedding;
- **point id** = `ClothingItem.id`;
- **payload** = every other field, as JSON.

Search combines two things:

1. **Payload filters for hard constraints.** `ItemFilter` (categories, occasions,
   formality range, warmth range, flowy) is translated by `build_filter()` into a Qdrant
   filter. A summer outfit never even considers a wool coat.
2. **Vector similarity for ranking** what is left.

Payload indexes are created on `category`, `formality`, `warmth`, `occasions` and
`flowy`. Local mode ignores them (it scans; fine for a personal wardrobe), but the same
code then runs unchanged against a Qdrant server. `ensure_collection()` also checks the
vector size, so a store built with a different CLIP model fails loudly instead of
returning nonsense.

`WardrobeStore` methods: `ensure_collection`, `upsert`, `update_tags` (payload only,
the vector stays), `delete`, `get`, `vector`, `search`, `list_items`, `count`.

## Environment guardrails

- **torch is never touched.** The shared venv has torch
  2.7.0+cu118 pinned. `open_clip_torch` needs torch, but torch is deliberately left out
  of `pyproject.toml`, and any new dependency is dry-run (`pip install --dry-run`)
  first to make sure it doesn't pull a different build. All local inference runs on
  CPU.
- **The SSL fix.** The venv has `pip-system-certs`, which replaces `ssl.SSLContext` at
  interpreter start. The openai client's HTTP stack wraps its own copy of the same
  library, and the two recurse until `RecursionError`. `outfit/__init__.py` calls
  `undo_pip_system_certs()` before anything imports openai, so `outfit` must always be
  imported first.
- **Secrets.** The DeepSeek key lives only in `.env` (template: `.env.example`). A
  PreToolUse hook, deny rules and a `.ignore` file stop Claude Code from ever reading
  it.

## Testing

- `tests/test_schemas.py`: 1–5 scales, unknown enum values and extra fields rejected,
  JSON round trip, stable ids.
- `tests/test_qdrant_store.py`: an in-memory Qdrant (`QdrantClient(":memory:")`):
  upsert without duplicates, filters, ranking, tag updates that keep the vector, the
  vector-size check.
- `tests/test_ssl_fix.py`: importing `outfit` restores the stdlib `SSLContext`, and the
  fix does nothing when ssl isn't patched.
- `tests/factories.make_item(**overrides)` builds valid test items for every later
  phase.
