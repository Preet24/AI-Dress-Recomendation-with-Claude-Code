"""Typed client for the outfit recommender API, used by the Streamlit pages.

The UI never imports the server code (that would load torch, open Qdrant and break the
one-process rule); it only needs `outfit.schemas` for the tag enums. These response
models mirror the API's, and `tests/test_ui_client.py` parses real API responses with
them, so a change on either side fails a test instead of the page.
"""

from datetime import date, datetime, time
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from outfit.schemas import Category, Fit, Length, Occasion, Pattern

DEFAULT_URL = "http://127.0.0.1:8000"
Mode = Literal["flat", "worn"]  # SegMode values: a garment alone / worn by a person
# outfit.profile's enums (importing it would load Qdrant through outfit.tools)
Gender = Literal["man", "woman", "non_binary"]
Build = Literal["slim", "average", "athletic", "broad", "curvy", "plus_size"]
DressCode = Literal["formal", "business", "business_casual", "smart_casual", "casual"]
Transport = Literal["walk", "bike", "two_wheeler", "public_transit", "car"]


class APIError(RuntimeError):
    """The API refused or failed a request; `detail` is its message, meant for the user."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class Item(BaseModel):
    id: UUID
    name: str  # what the agents call it, e.g. "lavender skirt"
    image_url: str
    category: Category
    subtype: str
    primary_color_name: str
    secondary_colors: list[str]
    pattern: Pattern
    fabric: str
    length: Length
    fit: Fit
    formality: int
    warmth: int
    flowy: bool
    occasions: list[Occasion]


class Outfit(BaseModel):
    pieces: list[Item]
    reasons: list[str]
    warnings: list[str]


class ChatTurn(BaseModel):
    thread_id: str
    kind: Literal["reply", "question"]
    text: str
    outfits: list[Outfit]
    gaps: list[str]
    shown: list[Item]
    path: list[str]
    llm_calls: int
    seconds: float


class Conversation(BaseModel):
    thread_id: str
    messages: list[dict[str, str]]
    question: str | None


class Health(BaseModel):
    status: str
    items: int
    model: str


class SavedPlace(BaseModel):
    """Mirrors outfit.profile.SavedPlace; the API does the validating."""

    name: str
    occasion: Occasion
    dress_code: DressCode | None = None
    start: time | None = None
    hours_away: float | None = None
    hours_outdoors: float | None = None
    transport: Transport | None = None


class Colouring(BaseModel):
    """Mirrors outfit.profile.Colouring."""

    summary: str
    suits: list[str] = Field(default_factory=list)
    go_easy: list[str] = Field(default_factory=list)


class Profile(BaseModel):
    """Mirrors outfit.profile.Profile."""

    gender: Gender | None = None
    home_city: str | None = None
    height_cm: int | None = None
    build: Build | None = None
    waist: str | None = None
    top_size: str | None = None
    fit_tops: Fit | None = None
    fit_bottoms: Fit | None = None
    colors_liked: list[str] = Field(default_factory=list)
    colors_avoided: list[str] = Field(default_factory=list)
    places: list[SavedPlace] = Field(default_factory=list)
    colouring: Colouring | None = None
    occasions: list[Occasion] = Field(default_factory=list)


class Pairing(BaseModel):
    item: Item
    score: float  # color harmony, or CLIP similarity for a look-alike
    reason: str


class ShoppingReport(BaseModel):
    """Mirrors outfit.api.shopping.ReportOut."""

    candidate: Item  # image_url: the candidate's cutout
    verdict: Literal["buy", "maybe", "skip"]
    reasons: list[str]
    goes_with: list[Pairing]
    clashes: list[Pairing]
    color_note: str
    new_outfits: int
    unlocks: list[str]
    fixes: list[str]
    looks_like: Pairing | None
    owned: bool


class ShoppingCheck(BaseModel):
    reports: list[ShoppingReport]
    checked: list[str]
    climate: str


class Suggestion(BaseModel):
    """Mirrors outfit.api.shopping.SuggestionOut: a staple to buy, no picture."""

    name: str
    category: str
    verdict: Literal["buy", "maybe", "skip"]
    reasons: list[str]
    unlocks: list[str]
    fixes: list[str]
    new_outfits: int
    color_note: str
    goes_with: list[Pairing]
    together_with: str | None
    after: list[str]


class Gap(BaseModel):
    gap: str
    where: list[str]


class ShoppingSuggestions(BaseModel):
    no_outfit: list[str]
    gaps: list[Gap]
    suggestions: list[Suggestion]
    checked: list[str]
    climate: str


class Activity(BaseModel):
    """Mirrors outfit.tools.packing.Activity: one part of a trip day."""

    occasion: Occasion = Occasion.CASUAL
    start: time = time(10, 0)
    hours_away: float = 8
    hours_outdoors: float = 3
    transport: Transport = "public_transit"


class TripEvent(Activity):
    day: int  # 1 = the first day


class Wears(BaseModel):
    """Re-wear limits, in days."""

    tops: int = 2
    bottoms: int = 3
    one_pieces: int = 1


class PlacePick(BaseModel):
    """A place the city name could be; send the chosen one back as `Trip.place`."""

    label: str  # "Manali, Himachal Pradesh, India"
    name: str
    admin1: str
    country: str
    latitude: float
    longitude: float
    timezone: str


class Trip(BaseModel):
    """Mirrors outfit.tools.packing.Trip (the API validates it)."""

    city: str | None = None  # None: the profile's home city
    place: PlacePick | None = None  # the confirmed place; packing needs it
    first_day: date
    days: int
    everyday: Activity = Field(default_factory=Activity)
    events: list[TripEvent] = Field(default_factory=list)
    travel_days: bool = True
    evenings_in: bool = False  # a lounge slot (relax / sleep) every night of the stay
    wears: Wears = Field(default_factory=Wears)


class TripSlot(BaseModel):
    day: int
    name: str  # "day 2 party", "day 2 evening in"
    occasion: Occasion
    start: datetime
    hours_away: float
    hours_outdoors: float
    transport: Transport
    indoors: bool = False


class TripRow(BaseModel):
    slot: TripSlot
    weather: str  # "feels 11-18 °C, rain up to 22%, ..."
    source: Literal["forecast", "last year", "not checked", "indoors"]


class TripPreview(BaseModel):
    """Mirrors outfit.api.packing.PreviewOut: what the user confirms."""

    city: str | None
    matches: list[PlacePick]
    place: PlacePick | None
    dates: str
    rows: list[TripRow]
    wears: Wears
    assumptions: list[str]


class PackedPiece(BaseModel):
    item: Item
    days: list[int]


class TripDay(BaseModel):
    slot: TripSlot
    outfit: Outfit | None
    notes: list[str]
    weather: str


class PackingPlan(BaseModel):
    """Mirrors outfit.api.packing.PlanOut."""

    place: str | None
    dates: str
    pieces: list[PackedPiece]
    days: list[TripDay]
    gaps: list[str]
    notes: list[str]
    assumptions: list[str]


# outfit.wear_log's enums (mirrored like the profile's; a test checks they match)
Rating = Literal["loved", "fine", "not_again"]
Issue = Literal["too_warm", "too_cold", "too_formal", "too_casual", "colours", "uncomfortable"]
Verdict = Literal["yours", "unsure", "extra", "unknown"]
WearSource = Literal["chat", "packing", "other"]


class GarmentMatch(BaseModel):
    region: Literal["upper", "lower", "full"]
    verdict: Verdict
    logged: UUID | None
    best: UUID | None
    score: float
    logged_score: float | None
    message: str


class PhotoMatch(BaseModel):
    garments: list[GarmentMatch]
    unchecked: list[UUID]


class LookScore(BaseModel):
    score: int
    reason: str


class LookScores(BaseModel):
    """Mirrors outfit.wear_log.LookScores: the outfit as worn, never the person."""

    fit: LookScore
    proportion: LookScore
    colour: LookScore
    coordination: LookScore
    occasion: LookScore
    polish: LookScore
    tips: list[str]


class WornEntry(BaseModel):
    """Mirrors outfit.api.worn.WornOut: one logged outfit."""

    id: UUID
    worn_on: date
    occasion: Occasion | None
    rating: Rating
    issues: list[Issue]
    note: str
    felt: tuple[float, float] | None
    source: WearSource
    created_at: datetime
    pieces: list[Item]
    removed: int
    photo_url: str | None
    match: PhotoMatch | None
    look: LookScores | None


class OutfitAPI:
    """One method per endpoint. Pass `http` to reuse a client (tests pass a TestClient)."""

    def __init__(self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None) -> None:
        # Chat turns and photo uploads wait on DeepSeek and CPU models: allow minutes.
        self.http = http or httpx.Client(base_url=base_url, timeout=180)

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TransportError as err:
            raise APIError(0, f"Can't reach the API ({err}); is scripts/serve.py running?") from err
        if response.is_error:
            raise APIError(response.status_code, error_detail(response))
        return response

    def health(self) -> Health:
        return Health.model_validate(self._send("GET", "/health").json())

    def items(
        self, categories: list[Category] | None = None, occasions: list[Occasion] | None = None
    ) -> list[Item]:
        params = {"category": categories or [], "occasion": occasions or []}
        return [Item.model_validate(i) for i in self._send("GET", "/items", params=params).json()]

    def image(self, url: str) -> bytes:
        """The PNG at an item's `image_url`."""
        return self._send("GET", url).content

    def upload(self, filename: str, data: bytes, mode: Mode) -> list[Item]:
        response = self._send(
            "POST", "/items", files={"photo": (filename, data)}, data={"mode": mode}
        )
        return [Item.model_validate(i) for i in response.json()]

    def edit(self, item_id: UUID, changes: dict[str, Any]) -> Item:
        return Item.model_validate(self._send("PATCH", f"/items/{item_id}", json=changes).json())

    def retag(self, item_id: UUID, note: str) -> Item:
        response = self._send("POST", f"/items/{item_id}/retag", json={"note": note})
        return Item.model_validate(response.json())

    def delete(self, item_id: UUID) -> None:
        self._send("DELETE", f"/items/{item_id}")

    def chat(self, message: str, thread_id: str | None = None) -> ChatTurn:
        body = {"message": message} | ({"thread_id": thread_id} if thread_id else {})
        return ChatTurn.model_validate(self._send("POST", "/chat", json=body).json())

    def conversation(self, thread_id: str) -> Conversation:
        return Conversation.model_validate(self._send("GET", f"/chat/{thread_id}").json())

    def check_item(self, filename: str, data: bytes, mode: Mode = "worn") -> ShoppingCheck:
        """Describe a garment you might buy (one DeepSeek call per garment); stores nothing."""
        response = self._send(
            "POST", "/shopping/check", files={"photo": (filename, data)}, data={"mode": mode}
        )
        return ShoppingCheck.model_validate(response.json())

    def suggest_purchases(self, occasions: list[str] | None = None) -> ShoppingSuggestions:
        """What to buy for these occasions (none: the profile's); local, no DeepSeek call."""
        response = self._send("POST", "/shopping/suggest", json={"occasions": occasions or []})
        return ShoppingSuggestions.model_validate(response.json())

    def add_candidate(self, item_id: UUID) -> Item:
        response = self._send("POST", f"/shopping/candidates/{item_id}/add")
        return Item.model_validate(response.json())

    def preview_trip(self, trip: Trip) -> TripPreview:
        """The trip to confirm: places, dates, weather per day; no wardrobe, no DeepSeek."""
        body = trip.model_dump(mode="json")
        return TripPreview.model_validate(self._send("POST", "/packing/preview", json=body).json())

    def pack_trip(self, trip: Trip) -> PackingPlan:
        """Pack for a confirmed trip (`trip.place` set when it has a city)."""
        body = trip.model_dump(mode="json")
        return PackingPlan.model_validate(self._send("POST", "/packing", json=body).json())

    def profile(self) -> Profile:
        return Profile.model_validate(self._send("GET", "/profile").json())

    def save_profile(self, profile: Profile) -> Profile:
        body = profile.model_dump(mode="json")
        return Profile.model_validate(self._send("PUT", "/profile", json=body).json())

    def upload_profile_photo(self, filename: str, data: bytes) -> None:
        self._send("PUT", "/profile/photo", files={"photo": (filename, data)})

    def profile_photo(self) -> bytes | None:
        """The kept photo (JPEG), or None if there is none."""
        try:
            return self._send("GET", "/profile/photo").content
        except APIError as err:
            if err.status == 404:
                return None
            raise

    def delete_profile_photo(self) -> None:
        self._send("DELETE", "/profile/photo")

    def analyse_colouring(self) -> Colouring:
        """Sends the photo to DeepSeek (one call) and saves the result in the profile."""
        return Colouring.model_validate(self._send("POST", "/profile/analyse").json())

    def log_worn(
        self,
        pieces: list[UUID],
        rating: Rating,
        issues: list[Issue] | None = None,
        note: str = "",
        worn_on: date | None = None,
        occasion: Occasion | None = None,
        source: WearSource = "other",
    ) -> WornEntry:
        """I wore this (free, no DeepSeek)."""
        body = {
            "pieces": [str(p) for p in pieces],
            "rating": rating,
            "issues": issues or [],
            "note": note,
            "worn_on": worn_on.isoformat() if worn_on else None,
            "occasion": occasion,
            "source": source,
        }
        return WornEntry.model_validate(self._send("POST", "/worn", json=body).json())

    def worn(self, limit: int = 100) -> list[WornEntry]:
        """Logged outfits, newest first."""
        response = self._send("GET", "/worn", params={"limit": limit})
        return [WornEntry.model_validate(e) for e in response.json()]

    def correct_worn(self, entry_id: UUID, pieces: list[UUID]) -> WornEntry:
        """Set what was actually worn; the API checks the photo again."""
        body = {"pieces": [str(p) for p in pieces]}
        return WornEntry.model_validate(self._send("PATCH", f"/worn/{entry_id}", json=body).json())

    def delete_worn(self, entry_id: UUID) -> None:
        self._send("DELETE", f"/worn/{entry_id}")

    def upload_worn_photo(self, entry_id: UUID, filename: str, data: bytes) -> WornEntry:
        """Keep a photo of you wearing it and match it locally (CPU, nothing sent)."""
        response = self._send("PUT", f"/worn/{entry_id}/photo", files={"photo": (filename, data)})
        return WornEntry.model_validate(response.json())

    def delete_worn_photo(self, entry_id: UUID) -> None:
        self._send("DELETE", f"/worn/{entry_id}/photo")

    def ask_look(self, entry_id: UUID) -> LookScores:
        """Sends the photo to DeepSeek (one call); the scores are kept on the entry."""
        return LookScores.model_validate(self._send("POST", f"/worn/{entry_id}/look").json())


def error_detail(response: httpx.Response) -> str:
    """The API's message: our errors carry `detail` as text, FastAPI's 422s as a list."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(detail, list):  # request validation: [{"loc": [...], "msg": ...}, ...]
        return "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in detail if "loc" in e)
    return str(detail)
