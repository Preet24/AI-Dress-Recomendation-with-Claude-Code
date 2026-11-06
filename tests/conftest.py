import pytest

from outfit.schemas import ClothingItem
from tests.factories import make_item


@pytest.fixture
def item() -> ClothingItem:
    return make_item()
