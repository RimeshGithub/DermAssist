"""Loads data/disease_info.json once. All wording lives in the JSON file, not in code."""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from . import config

log = logging.getLogger("dermassist.info")

_SECTION_KEYS = ("description", "causes", "risk_factors", "characteristics",
                 "prevention", "seek_care", "additional", "sources")


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(config.DISEASE_INFO_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def general_sources() -> list[dict]:
    return _load().get("general_sources", [])


def get_info(class_code: str) -> dict:
    """Return info for a class code. If a class has no entry, return an explicit 'no information'
    record instead of inventing content."""
    entry = _load().get("classes", {}).get(class_code)
    if entry is None:
        log.warning("No disease information for class '%s'", class_code)
        return {
            "display_name": class_code,
            "category": "No information available",
            "short_description": "No educational information has been added for this category.",
            "description": "", "causes": [], "risk_factors": [], "characteristics": [],
            "prevention": [], "seek_care": ["If you have a skin concern, see a doctor or dermatologist."],
            "additional": "", "sources": [],
        }
    return {"display_name": entry["display_name"], "category": entry.get("category", ""),
            "short_description": entry.get("short_description", ""),
            **{k: entry.get(k, [] if k not in ("description", "additional") else "") for k in _SECTION_KEYS}}
