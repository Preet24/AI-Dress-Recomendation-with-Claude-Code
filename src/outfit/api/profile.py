"""Profile endpoints: read it, replace it, and the optional photo with its colour analysis.

The photo stays on this machine; only `POST /profile/analyse` sends a copy to DeepSeek.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, UploadFile

from outfit.api.errors import NotFound
from outfit.api.items import photo_suffix
from outfit.api.services import Services, get_services
from outfit.colouring import analyse_colouring
from outfit.profile import Colouring, Profile

router = APIRouter(prefix="/profile", tags=["profile"])
ServicesDep = Annotated[Services, Depends(get_services)]


def stored_photo(s: Services) -> bytes:
    photo = s.profile.photo()
    if photo is None:
        raise NotFound("No profile photo; upload one first")
    return photo


@router.get("")
def read_profile(s: ServicesDep) -> Profile:
    """The saved profile; every field is empty until the user fills it in."""
    return s.profile.load()


@router.put("")
def replace_profile(profile: Profile, s: ServicesDep) -> Profile:
    """Save the whole profile (send every field you want to keep)."""
    return s.profile.save(profile)


@router.put("/photo", status_code=204)
def upload_photo(photo: UploadFile, s: ServicesDep) -> Response:
    """Keep a photo of yourself, re-encoded without EXIF; nothing is sent anywhere."""
    max_mb = s.settings.max_upload_mb
    data = photo.file.read(int(max_mb * 1024 * 1024) + 1)
    photo_suffix(data, max_mb)  # checks it really is a JPEG / PNG / WEBP under the limit
    s.profile.save_photo(data)
    return Response(status_code=204)


@router.get("/photo")
def read_photo(s: ServicesDep) -> Response:
    return Response(stored_photo(s), media_type="image/jpeg")


@router.delete("/photo", status_code=204)
def delete_photo(s: ServicesDep) -> Response:
    """Remove the photo; a colouring analysis already saved is kept."""
    if not s.profile.delete_photo():
        raise NotFound("No profile photo")
    return Response(status_code=204)


@router.post("/analyse")
def analyse(s: ServicesDep) -> Colouring:
    """Send the photo to DeepSeek once (1 call) and save the colouring it describes."""
    colouring = analyse_colouring(stored_photo(s), s.client, s.settings)
    s.profile.save(s.profile.load().model_copy(update={"colouring": colouring}))
    return colouring
