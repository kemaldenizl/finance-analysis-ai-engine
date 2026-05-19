import re

import pycountry


TR_MONTHS = {
    "ocak": 1,
    "şubat": 2,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "agustos": 8,
    "eylül": 9,
    "eylul": 9,
    "ekim": 10,
    "kasım": 11,
    "kasim": 11,
    "aralık": 12,
    "aralik": 12,
}


ISO_CURRENCY_CODES = sorted(
    {
        currency.alpha_3
        for currency in pycountry.currencies
        if hasattr(currency, "alpha_3")
    },
    key=len,
    reverse=True,
)


CURRENCY_SYMBOL_ALIASES = {
    "₺": "TRY",
    "TL": "TRY",
    "TRY": "TRY",
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "¥": "JPY",
    "JPY": "JPY",
    "₽": "RUB",
    "RUB": "RUB",
    "₩": "KRW",
    "KRW": "KRW",
    "₹": "INR",
    "INR": "INR",
    "₴": "UAH",
    "UAH": "UAH",
    "₦": "NGN",
    "NGN": "NGN",
    "₫": "VND",
    "VND": "VND",
    "₪": "ILS",
    "ILS": "ILS",
    "₱": "PHP",
    "PHP": "PHP",
    "฿": "THB",
    "THB": "THB",
    "₡": "CRC",
    "CRC": "CRC",
    "₲": "PYG",
    "PYG": "PYG",
    "₸": "KZT",
    "KZT": "KZT",
    "د.إ": "AED",
    "AED": "AED",
    "ر.س": "SAR",
    "SAR": "SAR",
}


CURRENCY_ALIASES = {
    **{code: code for code in ISO_CURRENCY_CODES},
    **CURRENCY_SYMBOL_ALIASES,
}


CURRENCY_TOKENS = sorted(
    set(ISO_CURRENCY_CODES) | set(CURRENCY_SYMBOL_ALIASES.keys()),
    key=len,
    reverse=True,
)


CURRENCY_REGEX_TEXT = "(" + "|".join(re.escape(token) for token in CURRENCY_TOKENS) + ")"


DATE_REGEXES = [
    r"(?P<date>\d{1,2}\s+(?:Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik)\s+\d{4})",
    r"(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
    r"(?P<date>\d{4}[./-]\d{1,2}[./-]\d{1,2})",
]


MONEY_REGEX = re.compile(
    rf"""
    (?P<cur_before>{CURRENCY_REGEX_TEXT})?
    \s*
    (?P<amount>
        [-+]?
        \d{{1,3}}(?:[.,]\d{{3}})*(?:[.,]\d{{2}})
        |
        [-+]?
        \d+(?:[.,]\d{{2}})
    )
    \s*
    (?P<cur_after>{CURRENCY_REGEX_TEXT})?
    """,
    re.IGNORECASE | re.VERBOSE,
)


INSTALLMENT_TEXT_REGEX = re.compile(
    r"(?P<current>\d{1,2})\s*\.?\s*Taksit",
    re.IGNORECASE,
)


INSTALLMENT_FRACTION_REGEX = re.compile(
    r"\((?P<current>\d{1,2})\s*/\s*(?P<total>\d{1,2})\)",
    re.IGNORECASE,
)


INSTALLMENT_MULTIPLY_REGEX = re.compile(
    r"(?P<unit>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))\s*x\s*(?P<total>\d{1,2})\s*=\s*(?P<sum>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))",
    re.IGNORECASE,
)


NOISE_KEYWORDS = [
    "müşteri limiti",
    "musteri limiti",
    "kart limiti",
    "nakit avans limiti",
    "dönem borcunuz",
    "donem borcunuz",
    "min. ödeme",
    "min. odeme",
    "asgari ödeme",
    "asgari odeme",
    "toplam bonus",
    "harcanabilecek",
    "hesap özeti",
    "hesap ozeti",
    "son ödeme",
    "son odeme",
    "ekstre no",
    "ekstre sayfası",
    "mersis numarası",
    "büyük mükellefler",
    "garanti bankası",
    "müşteri numarası",
    "musteri numarası",
    "kart numarası",
    "kart numarasi",
]


CREDIT_WORDS = [
    "iade",
    "ödemeniz için teşekkür",
    "odemeniz icin tesekkur",
    "teşekkür ederiz",
    "tesekkur ederiz",
    "alacak",
    "refund",
    "cashback",
    "ödeme",
    "odeme",
]