"""Shopping endpoints: what should I buy, check a garment you might buy, add it if bought.

A check describes the photo like an upload (cut out, tagged, embedded: one DeepSeek
call per garment) but stores nothing; the report comes from `tools/shopping.py` over
the wardrobe, the profile and the home city's typical weather. Suggestions come from
`tools/staples.py` over the same inputs, with no LLM at all.
"""

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from outfit.api.errors import NotFound
from outfit.api.items import ItemOut, photo_path, photo_suffix
from outfit.api.services import Services, get_services
from outfit.db.qdrant_store import ItemFilter
from outfit.ingestion.background import SegMode
from outfit.ingestion.candidates import Candidate
from outfit.profile import Profile
from outfit.schemas import Category, Occasion
from outfit.tools.outfits import OutfitRequest, label
from outfit.tools.shopping import (
    Pairing,
    ShoppingReport,
    Situation,
    Verdict,
    accessory_gaps,
    baseline,
    check_candidate,
    situations,
    where,
)
from outfit.tools.staples import Suggestion, missing, suggest_purchases

router = APIRouter(prefix="/shopping", tags=["shopping"])
ServicesDep = Annotated[Services, Depends(get_services)]
LOOK_ALIKES = 3  # same-category neighbours fetched per candidate


class PairingOut(BaseModel):
    item: ItemOut
    score: float
    reason: str

    @classmethod
    def of(cls, pairing: Pairing) -> "PairingOut":
        return cls(item=ItemOut.of(pairing.item), score=pairing.score, reason=pairing.reason)


class ReportOut(BaseModel):
    candidate: ItemOut  # image_url points at the candidate's cutout
    verdict: Verdict
    reasons: list[str]
    goes_with: list[PairingOut]
    clashes: list[PairingOut]
    color_note: str
    new_outfits: int
    unlocks: list[str]
    fixes: list[str]
    looks_like: PairingOut | None
    owned: bool

    @classmethod
    def of(cls, report: ShoppingReport) -> "ReportOut":
        item = report.candidate
        return cls(
            candidate=ItemOut.of(item).model_copy(
                update={"image_url": f"/shopping/candidates/{item.id}/image"}
            ),
            verdict=report.verdict,
            reasons=report.reasons,
            goes_with=[PairingOut.of(p) for p in report.goes_with],
            clashes=[PairingOut.of(p) for p in report.clashes],
            color_note=report.color_note,
            new_outfits=report.new_outfits,
            unlocks=report.unlocks,
            fixes=report.fixes,
            looks_like=PairingOut.of(report.looks_like) if report.looks_like else None,
            owned=report.owned,
        )


class CheckOut(BaseModel):
    reports: list[ReportOut]  # one per garment found in the photo
    checked: list[str]  # "Office on hot days", ...
    climate: str  # how the weather kinds were chosen


class SuggestIn(BaseModel):
    occasions: list[Occasion] = []  # empty: the profile's occasions and saved places


class SuggestionOut(BaseModel):
    """A staple to buy: there is no picture, only its tags and why."""

    name: str  # "navy wool coat"
    category: Category
    verdict: Verdict
    reasons: list[str]
    unlocks: list[str]
    fixes: list[str]
    new_outfits: int
    color_note: str
    goes_with: list[PairingOut]  # the user's own pieces
    together_with: str | None  # the other half of a top + bottom pair
    after: list[str]  # earlier suggestions it was scored on top of

    @classmethod
    def of(cls, suggestion: Suggestion) -> "SuggestionOut":
        report = suggestion.report
        return cls(
            name=label(report.candidate),
            category=report.candidate.category,
            verdict=report.verdict,
            reasons=report.reasons,
            unlocks=report.unlocks,
            fixes=report.fixes,
            new_outfits=report.new_outfits,
            color_note=report.color_note,
            goes_with=[PairingOut.of(p) for p in report.goes_with],
            together_with=suggestion.together_with,
            after=suggestion.after,
        )


class GapOut(BaseModel):
    gap: str  # "no footwear suits office"
    where: list[str]  # "office in any weather"


class SuggestOut(BaseModel):
    no_outfit: list[str]  # situations with no good outfit, and the closest misses
    gaps: list[GapOut]
    suggestions: list[SuggestionOut]  # best first
    checked: list[str]
    climate: str


def get_candidate(s: Services, item_id: UUID) -> Candidate:
    candidate = s.candidates.get(item_id)
    if candidate is None:
        raise NotFound(f"No candidate {item_id}; check the photo again")
    return candidate


def today(s: Services) -> date:
    return (s.context.now or datetime.now()).date()


def profile_situations(
    profile: Profile, occasions: list[Occasion] | None = None
) -> list[Situation]:
    """What to check: the profile's occasions and all saved places, or only the chosen
    occasions (and the saved places for them, at their dress code)."""
    places = [
        (p.name, p.occasion, p.formality())
        for p in profile.places
        if not occasions or p.occasion in occasions
    ]
    return situations(occasions or profile.occasions, places)


def report_for(s: Services, candidate: Candidate) -> tuple[ShoppingReport, str]:
    profile = s.profile.load()
    kinds, climate = s.climate(profile.home_city, today(s))
    wardrobe = s.store.list_items()
    hits = s.store.search(
        candidate.vector, ItemFilter(categories=[candidate.item.category]), limit=LOOK_ALIKES
    )
    report = check_candidate(
        candidate.item,
        wardrobe,
        profile_situations(profile),
        kinds,
        look_alikes=[(h.item, h.score) for h in hits],
        taste=OutfitRequest(**profile.taste_fields()),
    )
    return report, climate


@router.post("/suggest")
def suggest(body: SuggestIn, s: ServicesDep) -> SuggestOut:
    """What should I buy? What the wardrobe is missing for the chosen occasions, and the
    staples that would help most. Local rules only: no DeepSeek call."""
    profile = s.profile.load()
    kinds, climate = s.climate(profile.home_city, today(s))
    wardrobe = s.store.list_items()
    checks = profile_situations(profile, body.occasions)
    taste = OutfitRequest(**profile.taste_fields())
    before = baseline(wardrobe, checks, kinds, taste)
    found = missing(before, kinds, accessory_gaps(wardrobe, before, taste))
    gender = profile.gender.value if profile.gender else None
    picks = suggest_purchases(wardrobe, checks, kinds, taste, gender)
    return SuggestOut(
        no_outfit=found.no_outfit,
        gaps=[GapOut(gap=gap, where=places) for gap, places in found.gaps.items()],
        suggestions=[SuggestionOut.of(p) for p in picks],
        checked=[where(c, k) for c in checks for k in kinds],
        climate=climate,
    )


@router.post("/check")
def check(
    photo: UploadFile, s: ServicesDep, mode: Annotated[SegMode, Form()] = SegMode.WORN
) -> CheckOut:
    """Should I buy this? `worn` = on a model or a person (default), `flat` = the garment
    alone. Nothing is added to the wardrobe."""
    max_mb = s.settings.max_upload_mb
    data = photo.file.read(int(max_mb * 1024 * 1024) + 1)
    suffix = photo_suffix(data, max_mb)
    folder = s.settings.candidates_dir
    with s.ingest_lock:
        path = photo_path(folder, photo.filename, data, suffix)
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            items, vectors = s.pipeline().describe_photo(path, mode, folder)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    candidates = s.candidates.put(items, vectors, path)
    results = [report_for(s, c) for c in candidates]
    return CheckOut(
        reports=[ReportOut.of(r) for r, _ in results],
        checked=results[0][0].situations if results else [],
        climate=results[0][1] if results else "",
    )


@router.get("/candidates/{item_id}/image", response_class=FileResponse)
def candidate_image(item_id: UUID, s: ServicesDep) -> FileResponse:
    path = get_candidate(s, item_id).item.image_path
    if not path.is_file():
        raise NotFound(f"The cutout of candidate {item_id} is missing")
    return FileResponse(path, media_type="image/png")


@router.post("/candidates/{item_id}/add", status_code=201)
def add_candidate(item_id: UUID, s: ServicesDep) -> ItemOut:
    """Bought it: store it in the wardrobe as tagged (no new DeepSeek call)."""
    get_candidate(s, item_id)
    with s.ingest_lock:
        item = s.candidates.add_to_wardrobe(
            item_id, s.store, s.settings.clean_dir, s.settings.raw_dir
        )
    if item is None:
        raise NotFound(f"No candidate {item_id}; check the photo again")
    return ItemOut.of(item)
