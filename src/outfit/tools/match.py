"""Match a user's words ("my lavender skirt", "the white tee") to wardrobe items by tags.

CLIP alone is not reliable enough for this on real photos: on the test wardrobe
"lavender skirt" scored the beige trousers above the lavender skirt (0.243 vs 0.226),
and "black sequin top" found the right item at 0.231, under the acceptance threshold.
The items' own tags already say "lavender skirt", so the words are compared first:

- the reference's head noun (its last word: "skirt", "top", "heels") must be one of
  the item's garment words (subtype or category), so a color alone never matches
  ("blue sweater" is not the blue jeans);
- matching items rank by how many of the reference's words their tags contain
  (color names, subtype, pattern, fabric, category).

Words are lowercased, a plural "s" is dropped ("jeans" = "jean", "heels" = "heel"),
and a hyphenated tag also matches its parts ("off-white" matches "white").
Callers break ties and handle "no match" (both fall back to CLIP).
"""

import re

from outfit.schemas import Category, ClothingItem, Pattern

WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
STOP_WORDS = {"my", "the", "a", "an", "that", "those", "this", "these", "one", "with", "and", "of"}
ALIASES = {"tee": "t-shirt", "tshirt": "t-shirt"}
CATEGORY_WORDS: dict[Category, set[str]] = {
    Category.TOP: {"top"},
    Category.BOTTOM: {"bottom"},
    Category.ONE_PIECE: {"one-piece"},
    Category.OUTERWEAR: {"outerwear", "jacket", "coat", "layer"},
    Category.FOOTWEAR: {"footwear", "shoe"},
    Category.ETHNIC_TOP: {"top"},
    Category.ETHNIC_BOTTOM: {"bottom"},
    Category.ACCESSORY: {"accessory"},
}


def stem(word: str) -> str:
    """Drop a plural "s" ("jeans" -> "jean"), but keep "dress" and short words."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def tokens(text: str) -> list[str]:
    """Content words of a phrase, in order: aliased, stemmed, without stop words."""
    words = (ALIASES.get(w, w) for w in WORD.findall(text.lower()))
    return [stem(w) for w in words if w not in STOP_WORDS]


def expand(words: list[str] | set[str]) -> set[str]:
    """The words plus the parts of hyphenated ones ("off-white" -> off-white, off, white)."""
    return {part for w in words for part in (w, *(stem(p) for p in w.split("-")))}


def garment_words(item: ClothingItem) -> set[str]:
    return expand(tokens(item.subtype)) | CATEGORY_WORDS[item.category]


def tag_words(item: ClothingItem) -> set[str]:
    text = " ".join(
        [item.primary_color_name, *item.secondary_colors, item.subtype, item.fabric]
        + ([item.pattern.value] if item.pattern is not Pattern.SOLID else [])
    )
    return expand(tokens(text)) | garment_words(item)


def matches_word(word: str, vocabulary: set[str]) -> bool:
    """A reference word matches if the tags have it, or every part of a hyphenated one."""
    return word in vocabulary or all(stem(p) in vocabulary for p in word.split("-"))


def tag_matches(reference: str, items: list[ClothingItem]) -> list[tuple[ClothingItem, int]]:
    """Items whose tags fit the reference, best first: (item, number of words matched).

    Empty when the reference names no garment the wardrobe has.
    """
    words = tokens(reference)
    if not words:
        return []
    head = words[-1]
    scored = [
        (item, sum(matches_word(w, tag_words(item)) for w in words))
        for item in items
        if matches_word(head, garment_words(item))
    ]
    return sorted(scored, key=lambda pair: -pair[1])


def best_matches(reference: str, items: list[ClothingItem]) -> list[ClothingItem]:
    """The items tied for the best tag match (several = ambiguous, none = no match)."""
    scored = tag_matches(reference, items)
    return [item for item, score in scored if score == scored[0][1]] if scored else []
