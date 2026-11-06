"""Domain errors -> HTTP status codes, mapped once for every endpoint."""

import logging
from functools import partial

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from outfit.agents.llm import LLMError
from outfit.colouring import NoFaceVisible
from outfit.ingestion.background import SegmentationError
from outfit.ingestion.review import EditError
from outfit.ingestion.tagger import TaggingError
from outfit.look import NoPersonVisible
from outfit.tools.packing import PlaceNotConfirmed

logger = logging.getLogger(__name__)


class NotFound(LookupError):
    """No item with that id (or prefix) in the wardrobe."""


class BadUpload(ValueError):
    """The uploaded file is not a usable photo (wrong type, too large, unreadable)."""


class Busy(RuntimeError):
    """The conversation is still answering an earlier message."""


class BadWearLog(ValueError):
    """A logged outfit can't be kept: unknown pieces, a future date, contradicting issues."""


ERROR_STATUS: dict[type[Exception], int] = {
    NotFound: 404,
    Busy: 409,
    BadUpload: 422,
    BadWearLog: 422,
    EditError: 422,  # an unknown field or invalid value in a tag edit
    SegmentationError: 422,  # no garment found in the photo
    NoFaceVisible: 422,  # the profile photo can't be judged for colouring
    NoPersonVisible: 422,  # nobody wearing the outfit in a wear-log photo
    PlaceNotConfirmed: 422,  # packing for a city before the user confirmed which one
    TaggingError: 502,  # DeepSeek returned no usable tags
    LLMError: 502,  # DeepSeek returned no usable reply
}


async def error_response(status: int, request: Request, exc: Exception) -> JSONResponse:
    if status >= 500:
        logger.warning("%s %s failed: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=status, content={"detail": str(exc)})


def add_error_handlers(app: FastAPI) -> None:
    for exc_type, status in ERROR_STATUS.items():
        app.add_exception_handler(exc_type, partial(error_response, status))
