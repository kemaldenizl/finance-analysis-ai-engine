from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CategoryCandidate:
    category: str
    subcategory: str
    examples: tuple[str, ...]
    patterns: tuple[str, ...]


class CategoryTaxonomy:
    def __init__(self, candidates: list[CategoryCandidate]):
        self.candidates = candidates

    @property
    def allowed_categories(self) -> set[str]:
        return {candidate.category for candidate in self.candidates}

    def embedding_documents(self) -> list[str]:
        output = []

        for candidate in self.candidates:
            examples = ", ".join(candidate.examples)
            output.append(
                f"category={candidate.category}; "
                f"subcategory={candidate.subcategory}; "
                f"examples={examples}"
            )

        return output


@lru_cache
def load_taxonomy() -> CategoryTaxonomy:
    resource_path = (
        Path(__file__).resolve().parent
        / "resources"
        / "category_taxonomy.yaml"
    )

    with resource_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    candidates = []

    for category in raw["categories"]:
        category_id = category["id"]

        for subcategory in category.get("subcategories", []):
            candidates.append(
                CategoryCandidate(
                    category=category_id,
                    subcategory=subcategory["id"],
                    examples=tuple(subcategory.get("examples", [])),
                    patterns=tuple(subcategory.get("patterns", [])),
                )
            )

    return CategoryTaxonomy(candidates=candidates)