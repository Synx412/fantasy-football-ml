from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd


ENRICHABLE_COLUMNS = [
    "minutes",
    "appearances",
    "starts",
    "goals",
    "assists",
    "clean_sheets",
    "saves",
    "rating",
    "xg",
    "xa",
    "yellow_cards",
    "red_cards",
]


def _key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalise_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).strip().lower() for c in result.columns]
    if "team" in result.columns and "club" not in result.columns:
        result = result.rename(columns={"team": "club"})
    required = {"name", "club"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(
            "Public enrichment CSV must contain name and club columns. Missing: "
            + ", ".join(sorted(missing))
        )
    result["_name_key"] = result["name"].map(_key)
    result["_club_key"] = result["club"].map(_key)
    for column in ENRICHABLE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def merge_public_enrichment(players: pd.DataFrame, enrichment: pd.DataFrame) -> pd.DataFrame:
    """Merge offline FBref/Understat-style season totals into a live player pool.

    The live provider stays authoritative for identity, price, availability and fixtures.
    Public data fills or strengthens season-stat fields only.
    """
    left = players.copy().reset_index(drop=True)
    right = normalise_enrichment(enrichment)
    left["_name_key"] = left["name"].map(_key)
    left["_club_key"] = left["club"].map(_key)

    usable = [c for c in ENRICHABLE_COLUMNS if c in right.columns]
    if not usable:
        raise ValueError(
            "Public enrichment CSV has no usable statistic columns. Include fields such as "
            "minutes, starts, goals, assists, xg or xa."
        )
    right = (
        right[["_name_key", "_club_key", *usable]]
        .drop_duplicates(["_name_key", "_club_key"], keep="last")
    )
    merged = left.merge(right, on=["_name_key", "_club_key"], how="left", suffixes=("", "_public"))

    matched = np.zeros(len(merged), dtype=bool)
    source = merged.get("data_source", pd.Series("", index=merged.index)).fillna("").astype(str)
    api_football_source = source.str.contains("API-Football", case=False, regex=False)
    for column in usable:
        public_column = f"{column}_public"
        values = pd.to_numeric(merged[public_column], errors="coerce")
        current = pd.to_numeric(merged.get(column), errors="coerce")
        if column in {"xg", "xa"}:
            # API-Football's player endpoint uses local volume proxies in providers.py,
            # so real public xG/xA should replace those proxy values when available.
            use_public = values.notna() & (api_football_source | current.isna() | (current <= 0))
        else:
            # Live provider totals remain authoritative; public data fills missing fields.
            use_public = values.notna() & (current.isna() | (current <= 0))
        merged.loc[use_public, column] = values[use_public]
        matched |= values.notna().to_numpy()
        merged = merged.drop(columns=[public_column])

    merged["public_data_match"] = matched
    if "data_source" in merged.columns:
        merged.loc[matched, "data_source"] = (
            merged.loc[matched, "data_source"].fillna("").astype(str) + " + public enrichment"
        ).str.strip(" +")
    merged = merged.drop(columns=["_name_key", "_club_key"])
    return merged
