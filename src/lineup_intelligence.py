from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .providers import normalize_club, normalize_name, safe_float


STARTER_LABELS = {
    "starter", "start", "starting", "starting xi", "starting_xi", "predicted starter",
    "predicted_starter", "likely starter", "likely_starter", "yes", "1", "true",
}
BENCH_LABELS = {
    "bench", "sub", "substitute", "predicted bench", "predicted_bench", "not starting",
    "not_starting", "no", "0", "false",
}
OUT_LABELS = {"out", "not in squad", "not_in_squad", "unavailable", "injured", "suspended"}


def _status_vote(value: object) -> float | None:
    text = str(value or "").strip().lower()
    if text in STARTER_LABELS:
        return 1.0
    if text in BENCH_LABELS or text in OUT_LABELS:
        return 0.0
    if text in {"doubt", "doubtful", "50/50", "rotation", "uncertain"}:
        return 0.5
    return None


def build_lineup_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize either aggregated probabilities or one-row-per-source lineup votes.

    Supported aggregated format:
      name, club, start_probability, source_count[, expected_minutes]

    Supported vote format:
      source, name, club, status[, source_weight]
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "team" in df.columns and "club" not in df.columns:
        df = df.rename(columns={"team": "club"})
    if "player" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"player": "name"})
    if not {"name", "club"}.issubset(df.columns):
        raise ValueError("Lineup intelligence CSV must contain name and club/team columns.")

    df["_name_key"] = df["name"].map(normalize_name)
    df["_club_key"] = df["club"].map(normalize_club)

    probability_col = next(
        (c for c in ["start_probability", "starter_probability", "probability", "start_prob"] if c in df.columns),
        None,
    )
    if probability_col is not None:
        out = df[["name", "club", "_name_key", "_club_key"]].copy()
        out["lineup_consensus_probability"] = pd.to_numeric(df[probability_col], errors="coerce")
        # Accept either 0-1 or 0-100 input.
        over_one = out["lineup_consensus_probability"] > 1.0
        out.loc[over_one, "lineup_consensus_probability"] /= 100.0
        out["lineup_consensus_probability"] = out["lineup_consensus_probability"].clip(0.0, 1.0)
        if "source_count" in df.columns:
            out["lineup_source_count"] = pd.to_numeric(df["source_count"], errors="coerce").fillna(1.0).clip(1, 50)
        else:
            out["lineup_source_count"] = 1.0
        if "expected_minutes" in df.columns:
            out["lineup_expected_minutes"] = pd.to_numeric(df["expected_minutes"], errors="coerce").clip(0, 90)
        if "lineup_status" in df.columns:
            out["lineup_status"] = df["lineup_status"].fillna("").astype(str)
        out["lineup_intelligence_note"] = "uploaded aggregated lineup consensus"
        return out.drop_duplicates(["_name_key", "_club_key"], keep="last").reset_index(drop=True)

    if not {"source", "status"}.issubset(df.columns):
        raise ValueError(
            "Lineup intelligence CSV needs either start_probability/source_count or source/status columns."
        )

    df["_vote"] = df["status"].map(_status_vote)
    df = df[df["_vote"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["_weight"] = (
        pd.to_numeric(df.get("source_weight", 1.0), errors="coerce").fillna(1.0).clip(0.1, 5.0)
        if isinstance(df.get("source_weight", 1.0), pd.Series)
        else 1.0
    )
    df["_weighted_vote"] = df["_vote"] * df["_weight"]

    rows: list[dict] = []
    for (_, _), group in df.groupby(["_name_key", "_club_key"], sort=False):
        weight_sum = float(group["_weight"].sum())
        probability = float(group["_weighted_vote"].sum() / max(weight_sum, 1e-9))
        source_count = int(group["source"].astype(str).nunique())
        statuses = group["status"].fillna("").astype(str).str.lower()
        confirmed_status = ""
        if "confirmed" in group.columns:
            confirmed = group["confirmed"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
            if confirmed.any():
                confirmed_votes = group.loc[confirmed, "_vote"]
                if not confirmed_votes.empty:
                    confirmed_status = "starter" if float(confirmed_votes.iloc[-1]) >= 0.75 else "bench"
        if statuses.isin(OUT_LABELS).any() and (
            "confirmed" in group.columns
            and group["confirmed"].astype(str).str.lower().isin(["1", "true", "yes", "y"]).any()
        ):
            confirmed_status = "out"
        rows.append(
            {
                "name": group.iloc[-1]["name"],
                "club": group.iloc[-1]["club"],
                "_name_key": group.iloc[-1]["_name_key"],
                "_club_key": group.iloc[-1]["_club_key"],
                "lineup_consensus_probability": np.clip(probability, 0.0, 1.0),
                "lineup_source_count": float(source_count),
                "lineup_status": confirmed_status,
                "lineup_intelligence_note": f"{source_count} predicted-lineup source(s)",
            }
        )
    return pd.DataFrame(rows)


def merge_lineup_consensus(players: pd.DataFrame, intelligence: pd.DataFrame) -> pd.DataFrame:
    result = players.copy()
    if intelligence is None or intelligence.empty:
        return result
    info = build_lineup_consensus(intelligence) if "_name_key" not in intelligence.columns else intelligence.copy()
    if info.empty:
        return result

    result["_name_key"] = result["name"].map(normalize_name)
    result["_club_key"] = result["club"].map(normalize_club)
    lookup = info.drop_duplicates(["_name_key", "_club_key"], keep="last").set_index(["_name_key", "_club_key"])

    for idx, row in result.iterrows():
        key = (row["_name_key"], row["_club_key"])
        if key not in lookup.index:
            continue
        match = lookup.loc[key]
        result.at[idx, "lineup_consensus_probability"] = safe_float(match.get("lineup_consensus_probability"), np.nan)
        result.at[idx, "lineup_source_count"] = safe_float(match.get("lineup_source_count"), 1.0)
        if pd.notna(match.get("lineup_expected_minutes", np.nan)):
            result.at[idx, "lineup_expected_minutes"] = safe_float(match.get("lineup_expected_minutes"), np.nan)
        status = str(match.get("lineup_status") or "").strip().lower()
        if status:
            result.at[idx, "lineup_status"] = status
        result.at[idx, "lineup_intelligence_note"] = str(match.get("lineup_intelligence_note") or "")
    return result.drop(columns=["_name_key", "_club_key"])


def merge_confirmed_lineups(players: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Apply confirmed API lineups to every player on teams whose lineup is known."""
    result = players.copy()
    if lineups is None or lineups.empty:
        return result

    lineups = lineups.copy()
    lineups["_name_key"] = lineups["name"].map(normalize_name)
    lineups["_club_key"] = lineups["club"].map(normalize_club)
    result["_name_key"] = result["name"].map(normalize_name)
    result["_club_key"] = result["club"].map(normalize_club)

    confirmed_teams = set(lineups["_club_key"].dropna().astype(str))
    by_id = {}
    if "player_id" in lineups.columns:
        by_id = {
            int(row.player_id): str(row.lineup_status)
            for row in lineups.itertuples()
            if pd.notna(getattr(row, "player_id", np.nan))
        }
    by_key = {
        (str(row["_name_key"]), str(row["_club_key"])): str(row["lineup_status"])
        for _, row in lineups.iterrows()
    }

    for idx, row in result.iterrows():
        club_key = str(row["_club_key"])
        if club_key not in confirmed_teams:
            continue
        status = None
        player_id = row.get("player_id")
        if pd.notna(player_id):
            try:
                status = by_id.get(int(player_id))
            except (TypeError, ValueError):
                pass
        if status is None:
            status = by_key.get((str(row["_name_key"]), club_key))
        result.at[idx, "lineup_status"] = status or "not_in_squad"
        result.at[idx, "lineup_intelligence_note"] = "confirmed API-Football lineup"
        if status == "starter":
            result.at[idx, "lineup_consensus_probability"] = 1.0
            result.at[idx, "lineup_source_count"] = 99.0
        elif status == "bench":
            result.at[idx, "lineup_consensus_probability"] = 0.0
            result.at[idx, "lineup_source_count"] = 99.0
        elif status is None:
            result.at[idx, "lineup_consensus_probability"] = 0.0
            result.at[idx, "lineup_source_count"] = 99.0
    return result.drop(columns=["_name_key", "_club_key"])


def merge_recent_lineup_history(players: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Blend recent actual starts/minutes into the prior for every matched player."""
    result = players.copy()
    if history is None or history.empty:
        return result
    recent = history.copy()
    recent.columns = [str(c).strip().lower() for c in recent.columns]
    if "team" in recent.columns and "club" not in recent.columns:
        recent = recent.rename(columns={"team": "club"})
    if "player" in recent.columns and "name" not in recent.columns:
        recent = recent.rename(columns={"player": "name"})
    required = {"name", "club", "recent_start_rate"}
    missing = required - set(recent.columns)
    if missing:
        raise ValueError("Recent-lineup CSV is missing: " + ", ".join(sorted(missing)))

    recent["_name_key"] = recent["name"].map(normalize_name)
    recent["_club_key"] = recent["club"].map(normalize_club)
    recent["recent_start_rate"] = pd.to_numeric(recent["recent_start_rate"], errors="coerce").clip(0, 1)
    if "recent_lineup_matches" not in recent.columns:
        recent["recent_lineup_matches"] = 1.0
    recent["recent_lineup_matches"] = pd.to_numeric(
        recent["recent_lineup_matches"], errors="coerce"
    ).fillna(1.0).clip(1, 15)
    if "recent_minutes" in recent.columns:
        recent["recent_minutes"] = pd.to_numeric(recent["recent_minutes"], errors="coerce").clip(0, 90)

    lookup = recent.drop_duplicates(["_name_key", "_club_key"], keep="last").set_index(["_name_key", "_club_key"])
    result["_name_key"] = result["name"].map(normalize_name)
    result["_club_key"] = result["club"].map(normalize_club)
    if "start_probability" not in result.columns:
        result["start_probability"] = 0.5

    for idx, row in result.iterrows():
        key = (row["_name_key"], row["_club_key"])
        if key not in lookup.index:
            continue
        match = lookup.loc[key]
        rate = safe_float(match.get("recent_start_rate"), np.nan)
        if not np.isfinite(rate):
            continue
        sample = safe_float(match.get("recent_lineup_matches"), 1.0)
        weight = float(np.clip(0.15 + 0.08 * sample, 0.20, 0.65))
        prior = safe_float(row.get("start_probability"), 0.5)
        result.at[idx, "start_probability"] = float(np.clip((1.0 - weight) * prior + weight * rate, 0.0, 1.0))
        result.at[idx, "recent_start_rate"] = rate
        result.at[idx, "recent_lineup_matches"] = sample
        if pd.notna(match.get("recent_minutes", np.nan)):
            recent_minutes = safe_float(match.get("recent_minutes"), np.nan)
            result.at[idx, "recent_minutes"] = recent_minutes
            # Treat recent minutes as a soft minutes signal, not a confirmed lineup.
            if not pd.notna(row.get("lineup_expected_minutes", np.nan)):
                result.at[idx, "lineup_expected_minutes"] = recent_minutes
        previous_note = str(row.get("lineup_intelligence_note") or "").strip()
        note = f"recent actual lineups ({int(round(sample))} matches)"
        result.at[idx, "lineup_intelligence_note"] = f"{previous_note}; {note}".strip("; ")
    return result.drop(columns=["_name_key", "_club_key"])
