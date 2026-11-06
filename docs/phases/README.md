# Build phases

The project was built in phases, each one finished and tested before the next began.
Each file below explains one phase: what it does, why it is built that way, the code
that implements it, and how it is tested.

| Phase | What it delivers | Status |
|---|---|---|
| [1. Foundations](phase-1-foundations.md) | Data model, settings, vector store, environment guardrails | ✅ |
| [2. Ingestion](phase-2-ingestion.md) | Photo → cutout → tags → color → CLIP vector → Qdrant, plus a review tool | ✅ |
| [3. Tools](phase-3-tools.md) | Deterministic tools: wardrobe search, weather advice, color harmony, outfit assembly | ✅ |
| [4. Agents](phase-4-agents.md) | LangGraph multi-agent chat: router, intake with human-in-the-loop, ReAct refine, stylist, critic | ✅ steps 1–4 (step 5 became Phase 5) |
| [5. Agent evaluation](phase-5-evaluation.md) | Recorded conversations, a 21-scenario eval with per-agent metrics, fixes from the results | ✅ final prompts 63/63 over 3 runs |
| [6. API](phase-6-api.md) | FastAPI server: wardrobe, photo upload and agent chat over HTTP | ✅ |
| [7. UI](phase-7-ui.md) | Streamlit app: chat with outfit pictures, wardrobe browse / upload / edit, all through the API | ✅ (Docker left for later) |

Features built after the phases, on top of them: the user profile (taste, saved places,
home city; see [`../features_to_add.md`](../features_to_add.md)) and the
[shopping assistant](shopping-assistant.md) ("What should I buy?" and "Should I buy
this?", deterministic scoring over the Phase 3 tools), and
[accessories](accessories.md) (belts, ties, bags, sunglasses ... added to an outfit
after it is chosen, by plain rules), the [trip packing list](packing.md) (confirm
the place, dates and weather, then the fewest pieces that cover every day), and
[wear feedback](wear-feedback.md) ("I wore this": ratings nudge the ranking, a photo is
matched locally, an opt-in look check scores the outfit).

How the phases build on each other:

```
Phase 1  ClothingItem schema · Settings · WardrobeStore (Qdrant)
   │
Phase 2  photos ──► items + vectors in the store
   │
Phase 3  store ──► search · weather ──► advice · harmony ──► ranked OutfitPlan
   │
Phase 4  chat message ──► agents decide ──► Phase 3 tools ──► explained reply
   │
Phase 5  scripted conversations ──► real model, fixed inputs ──► scores per agent ──► fixes
   │
Phase 6  HTTP ──► one process owns store + models ──► ingestion · agents ──► JSON + images
   │
Phase 7  browser ──► Streamlit pages ──► API client only ──► chat + outfit pictures · wardrobe
```

The rule running through all of them: **code owns the facts and the rules, models
only do what needs perception or language.** The vision model names a garment but its
color is measured locally; the agents choose what to do next, but every outfit rule is
plain, unit-tested Python.

The detailed agent design (state, data flow per node, failure handling) is in
[`../agentic-architecture.md`](../agentic-architecture.md).
