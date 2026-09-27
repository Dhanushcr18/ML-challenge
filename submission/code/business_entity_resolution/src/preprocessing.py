"""Conservative, reusable text normalization."""
import re
import unicodedata
import pandas as pd

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_NAME_MAP = {"corporation": "corp", "corporate": "corp", "private": "pvt", "limited": "ltd"}
_ADDRESS_MAP = {"street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd", "lane": "ln", "drive": "dr", "apartment": "apt", "suite": "ste", "highway": "hwy", "roadway": "rd"}


def _base(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def normalize_name(value) -> str:
    return " ".join(_NAME_MAP.get(t, t) for t in _base(value).split())


def normalize_address(value) -> str:
    return " ".join(_ADDRESS_MAP.get(t, t) for t in _base(value).split())


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["business_name_normalized"] = out.business_name.map(normalize_name)
    out["business_address_normalized"] = out.business_address.map(normalize_address)
    out["country_normalized"] = out.country.map(_base)
    return out
