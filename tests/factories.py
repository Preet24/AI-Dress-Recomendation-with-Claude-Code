from pathlib import Path

from outfit.schemas import Category, ClothingItem, Fit, Length, Occasion, Pattern


def make_item(**overrides) -> ClothingItem:
    """A valid ClothingItem; override any field per test."""
    fields = {
        "image_path": Path("data/wardrobe_clean/shirt_whole.png"),
        "category": Category.TOP,
        "subtype": "oxford shirt",
        "primary_color_name": "light blue",
        "primary_color_lab": (78.0, -3.5, -15.2),
        "secondary_colors": ["white"],
        "pattern": Pattern.SOLID,
        "fabric": "cotton",
        "length": Length.FULL,
        "fit": Fit.REGULAR,
        "formality": 3,
        "warmth": 2,
        "flowy": False,
        "occasions": [Occasion.OFFICE, Occasion.CASUAL],
    }
    return ClothingItem(**(fields | overrides))
