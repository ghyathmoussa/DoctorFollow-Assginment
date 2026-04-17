"""Query preprocessing helpers for multilingual medical retrieval."""

from __future__ import annotations

from src.retrieval.corpus import normalize_text


TURKISH_QUERY_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("çocuklarda", "children pediatric"),
    ("çocuk", "child pediatric"),
    ("akut", "acute"),
    ("tedavisi", "treatment management therapy"),
    ("tedavi", "treatment therapy"),
    ("nasıl yapılır", "management treatment guideline"),
    ("tanı kriterleri", "diagnosis criteria diagnostic criteria"),
    ("tanı", "diagnosis diagnostic"),
    ("hastalığı", "disease"),
    ("hastaligi", "disease"),
    ("nelerdir", "criteria guideline"),
    ("çölyak", "celiac coeliac"),
    ("colyak", "celiac coeliac"),
)

TURKISH_MEDICAL_ALIASES: tuple[tuple[str, str], ...] = (
    ("akut otitis media", "acute otitis media"),
    ("çölyak hastalığı", "celiac disease coeliac disease"),
    ("colyak hastaligi", "celiac disease coeliac disease"),
    ("tip 2 diyabet", "type 2 diabetes mellitus"),
    ("demir eksikliği anemisi", "iron deficiency anemia"),
    ("demir eksikligi anemisi", "iron deficiency anemia"),
    ("toplum kökenli pnömoni", "community acquired pneumonia"),
    ("toplum kokenli pnomoni", "community acquired pneumonia"),
    ("gebelik diyabeti", "gestational diabetes"),
    ("alerjik rinit", "allergic rhinitis"),
)

TURKISH_CHARACTER_HINTS = frozenset("çğıöşü")


def _contains_turkish_signal(text: str) -> bool:
    lowered = text.casefold()
    if any(char in lowered for char in TURKISH_CHARACTER_HINTS):
        return True
    return any(alias in lowered for alias, _ in TURKISH_QUERY_EXPANSIONS + TURKISH_MEDICAL_ALIASES)


def expand_query(query: str, *, enable_translation: bool = True) -> str:
    """Append English retrieval hints for Turkish medical queries."""

    normalized_query = normalize_text(query)
    if not enable_translation or not normalized_query:
        return normalized_query

    lowered = normalized_query.casefold()
    if not _contains_turkish_signal(lowered):
        return normalized_query

    expansions: list[str] = []
    seen = {lowered}

    for alias, translation in TURKISH_MEDICAL_ALIASES:
        if alias in lowered and translation not in seen:
            expansions.append(translation)
            seen.add(translation)

    for phrase, expansion in TURKISH_QUERY_EXPANSIONS:
        if phrase in lowered and expansion not in seen:
            expansions.append(expansion)
            seen.add(expansion)

    if not expansions:
        return normalized_query

    return normalize_text(f"{normalized_query} {' '.join(expansions)}")
