# Phase 2 — Ingestion

Turns photos of your clothes into a searchable wardrobe: each garment is cut out,
described, measured and indexed.

```
photo ─► segment (rembg, CPU) ─► tag (DeepSeek vision) ─► color (k-means, LAB) ─► embed (CLIP, CPU) ─► Qdrant
```

## What it delivers

| Step | File | Runs where |
|---|---|---|
| Segmentation | `ingestion/background.py` | Local, ONNX on CPU |
| Tagging | `ingestion/tagger.py` + `prompts/tagger.md` | DeepSeek API (the only step that sends data out) |
| Color | `ingestion/color.py` | Local |
| Embedding | `ingestion/embedder.py` | Local, CLIP on CPU |
| Pipeline | `ingestion/pipeline.py` | Ties the steps together |
| Review | `ingestion/review.py` + `scripts/review_wardrobe.py` | Fix what the pipeline got wrong |

## 1. Segmentation: cut the garment out

`rembg` with open-source ONNX models on `onnxruntime` (CPU, no torch). Two modes,
because a garment on a bed and a garment on a person are different problems:

| Mode | Model | Output |
|---|---|---|
| `flat` (flat-lay, hanger, product shot) | `birefnet-general` | one `whole` cutout |
| `worn` (on a person) | `u2net_cloth_seg` | `upper` and `lower` cutouts, or one `full` for a one-piece |

So one photo can hold several garments, and `ingest_image()` returns a list.

**Choosing the flat model.** The first model, `isnet-general-use`, did poorly on real
phone photos of clothes laid on a bed: it lost the left half of a striped shirt, and it
kept the sheet showing between the legs of a pair of jeans. Five rembg models were
compared on the same 7 photos, all with the same mask cleaning:

| Model | Clean cutouts (of 7) | CPU time per photo |
|---|---|---|
| `isnet-general-use` (before) | 3 | 2.2 s |
| `birefnet-general-lite` | 2 | 12.6 s |
| **`birefnet-general`** | **7** (small bed scraps at the edge of 3) | 26.3 s |
| `bria-rmbg` (RMBG 2.0, non-commercial licence) | 4 | 30.3 s |
| `sam` (ViT-B, pointed at the photo's centre) | 3 | 14.1 s |

`birefnet-general` was the only one that cut the bed out between trouser legs. SAM
cuts out whatever is under the point it is given: on both pairs of trousers that was
the sheet between the legs, and on the red polo one letter of its print. Uploads now
take about half a minute; `OUTFIT_FLAT_MODEL=isnet-general-use` switches back.

Cleaning the masks (`clean_mask`):

- speckles are dropped, but blobs at least 20% the size of the largest are kept, so
  both shoes of a pair survive;
- tiny enclosed holes (< 0.1% of the image) are filled;
- in `flat` mode, parts thinner than about 1.6% of the long side are trimmed
  (`remove_thin_parts`: an opening finds the thick body, and the mask is kept only
  close to it). Every model keeps the hanger, because it lies on the garment; this
  removes the hook, while the bar across the collar stays. On 4 hanger photos it removed
  every hook and kept at least 99.8% of the garment. A mask with no thick part (a belt)
  is left as it is;
- in `worn` mode, the cloth model often labels the top half of a dress as "upper".
  When the "full" mask covers at least half of all garment pixels, all three masks are
  merged into one one-piece (`merge_worn_regions`);
- regions under `min_area_ratio` (2% of the image) are skipped.

Each cutout is an RGBA image cropped to the garment with a little padding, saved as
`data/wardrobe_clean/<photo>_<hash>_<region>.png`.

## 2. Tagging: describe the garment

DeepSeek (`deepseek-flash`, image input) through the `openai` SDK pointed at
`https://api.deepseek.com`; DeepSeek has no Python SDK, and nothing goes to OpenAI.

- **Only the cutout is sent, flattened onto grey.** Transparent pixels in a cutout
  still hold the old background, so sending the RGBA as-is would leak the room.
- **JSON mode + Pydantic.** JSON mode guarantees a JSON object, not the schema, so the
  schema is written into the system prompt (`{{schema}}` in `tagger.md`) and `TagResult`
  validates the reply. A unit test keeps the prompt's schema in sync with the model.
- **One retry.** Invalid, empty or truncated output is sent back with the error for
  one more try, then `TaggingError`. A `content_filter` refusal is not retried.
- The region ("upper", "whole"…) is passed as a **hint only**: the model decides the
  category.

## 3. Color: measured, not guessed

The model names the color ("lavender"), but the number used for color harmony is
computed locally:

- take the cutout's opaque pixels (subsampled to 20,000);
- convert RGB → CIELAB, where distance roughly matches how different colors look;
- k-means (k = 4), largest cluster first. Its center becomes `primary_color_lab`.

Clustering in LAB keeps a lit fold and a shadowed fold of the same shirt together
more often than RGB would.

## 4. Embedding: CLIP on CPU

OpenCLIP ViT-B-32 with `laion2b_s34b_b79k` weights, 512-dimensional, on CPU.

- Cutouts are composited onto the same neutral grey (128, 128, 128) before
  embedding, so the background never pulls a vector toward "dark" or "white".
- Vectors are L2-normalized, so cosine similarity is a dot product.
- Images and text share one space: a cutout can later be found by another cutout
  ("more like this") or by a phrase ("a light linen shirt").

## 5. The pipeline

`Pipeline.ingest_image(path, mode)`:

1. segment the photo into cutouts and save them;
2. tag and color every cutout, under its stable id;
3. embed all cutouts in one batch;
4. upsert everything into Qdrant.

**Nothing is written until every garment in the photo is tagged and embedded**, so a
failed photo leaves the store untouched. Re-ingesting a photo produces the same ids
and keeps the original `created_at`.

## 6. Review: fix what went wrong

Vision tagging is good, not perfect (a skirt tagged as shorts, a sequin top tagged as
plain). `scripts/review_wardrobe.py` walks the items one by one with these actions:

- **keep** the item;
- **replace** it from a photo (the same photo in the other mode, or a better one). The
  new photo is ingested first; the old item is removed only after that succeeds;
- **note**: re-tag the same cutout with the owner's correction ("this is a midi slip
  dress"), which the tagger is told to trust over the image;
- **fields**: edit tags directly, e.g. `category=one_piece length=midi`;
- **delete** the item and its cutout.

Only tag fields (`TagResult`) are editable. The id, cutout, vector, LAB color and
`created_at` belong to the pipeline.

## Commands

```bash
python scripts/ingest_folder.py data/wardrobe_raw --mode flat   # or --mode worn
python scripts/segment_folder.py data/wardrobe_raw --mode worn  # cutouts only, to check
python scripts/review_wardrobe.py [--match skirt] [--no-show]
```

## Testing

- The DeepSeek client is faked (`tests/fakes.FakeClient`, `tags_reply(**overrides)`);
  tests never call the real API.
- Segmentation tests use a fake `MaskSession`, so speckle removal, kept shoe pairs,
  filled pinholes and the dress-vs-top/bottom decision are checked without a model.
- `pytest -q -m slow` also runs the real ONNX and CLIP models on fixture images.
- Files: `test_background.py`, `test_tagger.py`, `test_color.py`, `test_embedder.py`,
  `test_pipeline.py`, `test_review.py`.
