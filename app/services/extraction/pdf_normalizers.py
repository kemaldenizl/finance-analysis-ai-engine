import re
import unicodedata
from datetime import datetime

from app.services.extraction.pdf_patterns import (
    CURRENCY_ALIASES,
    CURRENCY_REGEX_TEXT,
    ISO_CURRENCY_CODES,
    TR_MONTHS,
)


def normalize_text_for_matching(value: str) -> str:
    value = value or ""
    value = value.lower()
    value = value.replace("ı", "i")
    value = value.replace("\u00a0", " ")

    return unicodedata.normalize("NFKC", value)


def normalize_currency(value: str | None) -> str | None:
    if not value:
        return None

    value = str(value).strip()
    upper_value = value.upper()

    if upper_value in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[upper_value]

    if value in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[value]

    return upper_value if upper_value in ISO_CURRENCY_CODES else None


def parse_money_number(value: str | None) -> float | None:
    if value is None:
        return None

    value = str(value).strip()
    negative = False

    if value.startswith("-"):
        negative = True
        value = value[1:]

    if value.startswith("+"):
        value = value[1:]

    comma = value.rfind(",")
    dot = value.rfind(".")

    if comma != -1 and dot != -1:
        if comma > dot:
            value = value.replace(".", "")
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    elif comma != -1:
        value = value.replace(".", "")
        value = value.replace(",", ".")
    elif dot != -1:
        parts = value.split(".")

        if len(parts[-1]) == 2:
            value = value.replace(",", "")
        else:
            value = value.replace(".", "")

    try:
        number = float(value)

        return -number if negative else number

    except Exception:
        return None


def normalize_date(raw: str) -> str | None:
    raw = raw.strip()

    month_name_match = re.match(
        r"^(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4})$",
        raw,
        re.IGNORECASE,
    )

    if month_name_match:
        day = int(month_name_match.group(1))
        month_name = normalize_text_for_matching(month_name_match.group(2))
        year = int(month_name_match.group(3))

        month = TR_MONTHS.get(month_name)

        if month:
            return datetime(year, month, day).strftime("%Y-%m-%d")

    numeric_match = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$", raw)

    if numeric_match:
        day = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        year = int(numeric_match.group(3))

        if year < 100:
            year += 2000

        return datetime(year, month, day).strftime("%Y-%m-%d")

    iso_like_match = re.match(r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$", raw)

    if iso_like_match:
        year = int(iso_like_match.group(1))
        month = int(iso_like_match.group(2))
        day = int(iso_like_match.group(3))

        return datetime(year, month, day).strftime("%Y-%m-%d")

    return None


def strip_currency_tokens(value: str) -> str:
    return re.sub(CURRENCY_REGEX_TEXT, " ", value, flags=re.IGNORECASE)


def normalize_whitespace(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def remove_pdf_artifacts(value: str) -> str:
    value = value.replace("bosluk", " ")
    value = value.replace("BOŞLUK", " ")
    value = value.replace("□", " ")

    return normalize_whitespace(value)