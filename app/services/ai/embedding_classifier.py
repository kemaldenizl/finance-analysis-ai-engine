import logging
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

from app.core.config import settings
from app.services.ai.category_taxonomy import CategoryTaxonomy, load_taxonomy


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCategoryPrediction:
    category: str
    subcategory: str
    confidence: float
    similarity: float
    matched_reference: str


class EmbeddingCategoryClassifier:
    def __init__(self, taxonomy: CategoryTaxonomy | None = None):
        self.taxonomy = taxonomy or load_taxonomy()
        self._model: SentenceTransformer | None = None
        self._reference_documents: list[str] | None = None
        self._reference_embeddings = None

    def predict(self, merchant_text: str) -> EmbeddingCategoryPrediction | None:
        if not settings.EMBEDDING_ENABLED:
            return None

        if not merchant_text.strip():
            return None

        try:
            self._ensure_loaded()

            query_embedding = self._model.encode(
                [merchant_text],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            similarities = util.cos_sim(
                query_embedding,
                self._reference_embeddings,
            )[0]

            best_index = int(similarities.argmax())
            best_similarity = float(similarities[best_index])

            if best_similarity < settings.EMBEDDING_SIMILARITY_THRESHOLD:
                return None

            candidate = self.taxonomy.candidates[best_index]

            return EmbeddingCategoryPrediction(
                category=candidate.category,
                subcategory=candidate.subcategory,
                confidence=round(min(0.90, best_similarity), 4),
                similarity=round(best_similarity, 4),
                matched_reference=self._reference_documents[best_index],
            )

        except Exception as exc:
            logger.exception("Embedding classification failed: %s", exc)

            return None

    def is_available(self) -> bool:
        if not settings.EMBEDDING_ENABLED:
            return False

        try:
            self._ensure_loaded()

            return True

        except Exception as exc:
            logger.warning("Embedding model unavailable: %s", exc)

            return False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self._reference_documents = self.taxonomy.embedding_documents()

        self._reference_embeddings = self._model.encode(
            self._reference_documents,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )