import re
from collections import Counter

from app.schemas.extraction import ExtractedInstallment, ExtractedTransaction
from app.services.extraction.pdf_normalizers import (
    normalize_currency,
    normalize_date,
    normalize_text_for_matching,
    normalize_whitespace,
    parse_money_number,
    remove_pdf_artifacts,
    strip_currency_tokens,
)
from app.services.extraction.pdf_patterns import (
    CREDIT_WORDS,
    CURRENCY_REGEX_TEXT,
    DATE_REGEXES,
    INSTALLMENT_FRACTION_REGEX,
    INSTALLMENT_MULTIPLY_REGEX,
    INSTALLMENT_TEXT_REGEX,
    MONEY_REGEX,
    NOISE_KEYWORDS,
)
from app.services.extraction.pdf_text_extractor import PdfTextLine


class PdfTransactionParser:
    def build_candidate_lines(self, lines: list[PdfTextLine]) -> list[PdfTextLine]:
        """
        Native PDF'lerde bazı bankalar transaction satırını tek satır,
        bazıları iki/üç satır halinde verebilir.

        Bu yüzden:
        - Orijinal satırları koruyoruz.
        - Tarih var ama para yoksa aynı sayfadaki sonraki 1-2 satırla birleştiriyoruz.
        - Böylece layout değişikliklerine biraz daha tolerans sağlıyoruz.
        """
        candidates: list[PdfTextLine] = []
        seen: set[tuple[int, str]] = set()

        for index, line in enumerate(lines):
            self._append_unique(candidates, seen, line)

            if not self.find_date(line.text):
                continue

            if self.find_money_values(line.text):
                continue

            merged_text = line.text

            for lookahead in range(1, 3):
                next_index = index + lookahead

                if next_index >= len(lines):
                    break

                next_line = lines[next_index]

                if next_line.page != line.page:
                    break

                merged_text = normalize_whitespace(f"{merged_text} {next_line.text}")

                merged_line = PdfTextLine(
                    page=line.page,
                    text=merged_text,
                    source=f"{line.source}+merged",
                )

                self._append_unique(candidates, seen, merged_line)

                if self.find_money_values(merged_text):
                    break

        return candidates

    def _append_unique(
        self,
        candidates: list[PdfTextLine],
        seen: set[tuple[int, str]],
        line: PdfTextLine,
    ) -> None:
        key = (line.page, line.text)

        if key in seen:
            return

        seen.add(key)
        candidates.append(line)

    def infer_document_currency(self, lines: list[PdfTextLine]) -> str:
        joined = "\n".join(line.text for line in lines)

        currencies = []

        for raw in re.findall(CURRENCY_REGEX_TEXT, joined, re.IGNORECASE):
            normalized = normalize_currency(raw)

            if normalized:
                currencies.append(normalized)

        if not currencies:
            return "TRY"

        return Counter(currencies).most_common(1)[0][0]

    def find_date(self, text: str) -> dict | None:
        for pattern in DATE_REGEXES:
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                continue

            raw = match.group("date")

            try:
                normalized = normalize_date(raw)
            except Exception:
                normalized = None

            if normalized:
                return {
                    "raw": raw,
                    "date": normalized,
                    "start": match.start(),
                    "end": match.end(),
                }

        return None

    def find_money_values(self, text: str) -> list[dict]:
        values = []

        for match in MONEY_REGEX.finditer(text):
            raw_amount = match.group("amount")
            raw_currency = match.group("cur_before") or match.group("cur_after")

            amount = parse_money_number(raw_amount)
            currency = normalize_currency(raw_currency)

            if amount is None:
                continue

            values.append(
                {
                    "raw": match.group(0).strip(),
                    "amount": amount,
                    "currency": currency,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        return values

    def detect_installment(self, text: str) -> ExtractedInstallment:
        current = None
        total = None
        raw_parts: list[str] = []
        installment_total_amount = None
        installment_unit_amount = None

        multiply = INSTALLMENT_MULTIPLY_REGEX.search(text)

        if multiply:
            installment_unit_amount = parse_money_number(multiply.group("unit"))
            total = int(multiply.group("total"))
            installment_total_amount = parse_money_number(multiply.group("sum"))
            raw_parts.append(multiply.group(0))

        taksit = INSTALLMENT_TEXT_REGEX.search(text)

        if taksit:
            current = int(taksit.group("current"))
            raw_parts.append(taksit.group(0))

        fraction = INSTALLMENT_FRACTION_REGEX.search(text)

        if fraction:
            current = int(fraction.group("current"))
            total = int(fraction.group("total"))
            raw_parts.append(fraction.group(0))

        raw = " | ".join(raw_parts) if raw_parts else None

        return ExtractedInstallment(
            current=current,
            total=total,
            raw=raw,
            unit_amount=installment_unit_amount,
            total_amount=installment_total_amount,
        )

    def parse_line_as_transaction(
        self,
        line: PdfTextLine,
        document_currency: str,
    ) -> ExtractedTransaction | None:
        text = remove_pdf_artifacts(line.text)

        if self.is_noise_line(text):
            return None

        date_info = self.find_date(text)

        if not date_info:
            return None

        money_values = self.find_money_values(text)

        if not money_values:
            return None

        installment = self.detect_installment(text)

        price, currency, original_price, original_currency = self.choose_amounts(
            money_values=money_values,
            installment=installment,
            document_currency=document_currency,
        )

        if price is None:
            return None

        description = self.clean_description(
            text=text,
            date_raw=date_info["raw"],
            money_values=money_values,
            installment=installment,
        )

        if not description:
            return None

        confidence = self.calculate_confidence(
            date_found=True,
            price_found=price is not None,
            description=description,
            currency=currency,
            installment=installment,
        )

        return ExtractedTransaction(
            date=date_info["date"],
            description=description,
            price=abs(price),
            currency=currency or document_currency,
            original_price=abs(original_price) if original_price is not None else None,
            original_currency=original_currency,
            installment=installment,
            direction=self.infer_direction(text),
            confidence=confidence,
            page=line.page,
        )

    def is_noise_line(self, text: str) -> bool:
        normalized = normalize_text_for_matching(text)

        return any(keyword in normalized for keyword in NOISE_KEYWORDS)

    def clean_description(
        self,
        text: str,
        date_raw: str,
        money_values: list[dict],
        installment: ExtractedInstallment,
    ) -> str:
        description = text

        if date_raw:
            description = description.replace(date_raw, " ")

        for item in sorted(money_values, key=lambda value: len(value["raw"]), reverse=True):
            description = description.replace(item["raw"], " ")

        if installment.raw:
            for part in installment.raw.split("|"):
                description = description.replace(part.strip(), " ")

        description = strip_currency_tokens(description)
        description = remove_pdf_artifacts(description)
        description = description.strip(" -–—|:,;")

        return normalize_whitespace(description)

    def choose_amounts(
        self,
        money_values: list[dict],
        installment: ExtractedInstallment,
        document_currency: str,
    ) -> tuple[float | None, str | None, float | None, str | None]:
        if not money_values:
            return None, document_currency, None, None

        for money in money_values:
            if money["currency"] is None:
                money["currency"] = document_currency

        has_installment = installment.current is not None or installment.total is not None

        if has_installment:
            price_item = money_values[-1]

            return (
                price_item["amount"],
                price_item["currency"],
                installment.total_amount,
                price_item["currency"],
            )

        currencies = {money["currency"] for money in money_values if money["currency"]}

        if len(currencies) >= 2:
            price_item = None
            original_item = None

            for money in reversed(money_values):
                if money["currency"] == document_currency:
                    price_item = money
                    break

            if price_item is None:
                price_item = money_values[-1]

            for money in money_values:
                if money["currency"] != price_item["currency"]:
                    original_item = money
                    break

            return (
                price_item["amount"],
                price_item["currency"],
                original_item["amount"] if original_item else None,
                original_item["currency"] if original_item else None,
            )

        price_item = money_values[-1]

        return (
            price_item["amount"],
            price_item["currency"],
            None,
            None,
        )

    def infer_direction(self, text: str) -> str:
        normalized = normalize_text_for_matching(text)

        if any(word in normalized for word in CREDIT_WORDS):
            return "credit"

        return "debit"

    def calculate_confidence(
        self,
        date_found: bool,
        price_found: bool,
        description: str,
        currency: str | None,
        installment: ExtractedInstallment,
    ) -> float:
        confidence = 0.0

        confidence += 0.35 if date_found else 0.0
        confidence += 0.30 if price_found else 0.0
        confidence += 0.20 if len(description) >= 3 else 0.0
        confidence += 0.10 if currency else 0.0
        confidence += 0.03 if installment.current is not None else 0.0

        return round(min(confidence, 0.98), 2)