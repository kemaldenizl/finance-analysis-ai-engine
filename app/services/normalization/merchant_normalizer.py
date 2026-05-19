import re
import unicodedata

from app.schemas.normalization import NormalizedMerchant


class MerchantNormalizer:
    GENERIC_NOISE = [
        "taksit",
        "bonus",
        "kart",
        "ödeme",
        "odeme",
        "harcama",
        "işlem",
        "islem",
    ]

    KNOWN_ALIASES = {
        "OBILET": "OBILET.COM",
        "OBİLET": "OBILET.COM",
        "IYZICO AMAZON": "IYZICO *AMAZON.COM",
        "IYZICO *AMAZON": "IYZICO *AMAZON.COM",
        "BKMKITAP": "IYZICO/BKMKITAP.COM",
        "PEGASUS": "PEGASUS",
        "MEDIA MARKT": "MEDIA MARKT",
        "TRENDYOL": "TRENDYOL",
        "STEAM": "STEAM",
    }

    def normalize(self, description: str) -> NormalizedMerchant:
        raw = description or ""
        cleaned = self._clean(raw)
        normalized = self._canonicalize(cleaned)
        display_name = self._display_name(normalized)

        confidence = self._confidence(
            raw=raw,
            cleaned=cleaned,
            normalized=normalized,
        )

        return NormalizedMerchant(
            raw=raw,
            normalized=normalized,
            display_name=display_name,
            confidence=confidence,
        )

    def _clean(self, value: str) -> str:
        value = unicodedata.normalize("NFKC", value)
        value = value.replace("\u00a0", " ")
        value = value.replace("İ", "I")
        value = value.replace("ı", "i")
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" -–—|:,;")

        return value.strip()

    def _canonicalize(self, value: str) -> str:
        upper = value.upper()

        upper = re.sub(r"\s+", " ", upper)
        upper = re.sub(r"\s*-\s*", "-", upper)
        upper = upper.strip(" -–—|:,;")

        for alias, canonical in self.KNOWN_ALIASES.items():
            if alias in upper:
                return canonical

        upper = re.sub(r"\b\d{6,}\b", " ", upper)
        upper = re.sub(r"\s+", " ", upper).strip()

        return upper

    def _display_name(self, normalized: str) -> str:
        if not normalized:
            return "UNKNOWN"

        return normalized

    def _confidence(
        self,
        raw: str,
        cleaned: str,
        normalized: str,
    ) -> float:
        if not raw or not cleaned or not normalized:
            return 0.20

        score = 0.60

        if len(normalized) >= 3:
            score += 0.20

        if any(char.isalpha() for char in normalized):
            score += 0.10

        if normalized in self.KNOWN_ALIASES.values():
            score += 0.10

        return round(min(score, 0.98), 4)