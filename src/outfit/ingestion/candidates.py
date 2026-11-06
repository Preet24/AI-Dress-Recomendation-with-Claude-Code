"""Garments the user might buy: described like wardrobe items, but not stored yet.

"Check an item" runs `Pipeline.describe_photo` into `candidates_dir`; the results wait
here (in memory, lost on restart) until the user adds one to the wardrobe, which moves
its cutout and photo next to the wardrobe's and stores it with the vector already
computed, so it is not tagged again. Only the newest `MAX_CANDIDATES` are kept.
"""

import logging
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import UUID

import numpy as np

from outfit.db.qdrant_store import WardrobeStore
from outfit.schemas import ClothingItem

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 20


@dataclass(frozen=True)
class Candidate:
    item: ClothingItem
    vector: np.ndarray
    photo: Path  # the uploaded photo, shared by the garments found in it


class CandidateStore:
    def __init__(self, limit: int = MAX_CANDIDATES) -> None:
        self.limit = limit
        self._candidates: OrderedDict[UUID, Candidate] = OrderedDict()
        self._lock = Lock()

    def put(self, items: list[ClothingItem], vectors: np.ndarray, photo: Path) -> list[Candidate]:
        added = [Candidate(i, v, photo) for i, v in zip(items, vectors, strict=True)]
        with self._lock:
            for candidate in added:
                self._candidates[candidate.item.id] = candidate
                self._candidates.move_to_end(candidate.item.id)
            while len(self._candidates) > self.limit:
                _, old = self._candidates.popitem(last=False)
                self._remove_files(old)
        return added

    def get(self, item_id: UUID) -> Candidate | None:
        with self._lock:
            return self._candidates.get(item_id)

    def add_to_wardrobe(
        self, item_id: UUID, store: WardrobeStore, clean_dir: Path, raw_dir: Path
    ) -> ClothingItem | None:
        """Store the candidate in the wardrobe as it was tagged; None if it is unknown."""
        with self._lock:
            candidate = self._candidates.pop(item_id, None)
        if candidate is None:
            return None
        clean_dir.mkdir(parents=True, exist_ok=True)
        cutout = Path(
            shutil.move(candidate.item.image_path, clean_dir / candidate.item.image_path.name)
        )
        item = candidate.item.model_copy(update={"image_path": cutout})
        store.upsert([item], candidate.vector[None, :])
        if candidate.photo.is_file() and not self._photo_in_use(candidate.photo):
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(candidate.photo, raw_dir / candidate.photo.name)
        logger.info("Added candidate %s (%s) to the wardrobe", item.id, item.subtype)
        return item

    def _photo_in_use(self, photo: Path) -> bool:
        with self._lock:
            return any(c.photo == photo for c in self._candidates.values())

    def _remove_files(self, candidate: Candidate) -> None:
        candidate.item.image_path.unlink(missing_ok=True)
        if not any(c.photo == candidate.photo for c in self._candidates.values()):
            candidate.photo.unlink(missing_ok=True)
