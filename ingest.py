from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


EMAIL_COLUMNS = [
    "id",
    "thread_id",
    "thread_position",
    "timestamp",
    "from_name",
    "from_email",
    "from_type",
    "from_unit",
    "from_property_id",
    "from_role",
    "from_company",
    "to",
    "cc",
    "subject",
    "body",
    "attachments",
    "read",
]

PROPERTY_COLUMNS = [
    "property_id",
    "property_name",
    "property_type",
    "property_units",
    "property_manager",
]


def load_json(path: str) -> dict:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON dataset: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Dataset must be a top-level JSON object.")

    if not isinstance(raw.get("emails"), list):
        raise ValueError("Dataset must contain an 'emails' array.")

    if not isinstance(raw.get("metadata"), dict):
        raw["metadata"] = {}

    return raw


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def flatten_emails(raw: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for email in raw.get("emails", []):
        if not isinstance(email, dict):
            continue

        from_obj = email.get("from") or {}
        if not isinstance(from_obj, dict):
            from_obj = {}

        rows.append(
            {
                "id": email.get("id"),
                "thread_id": email.get("thread_id"),
                "thread_position": email.get("thread_position"),
                "timestamp": email.get("timestamp"),
                "from_name": from_obj.get("name"),
                "from_email": from_obj.get("email"),
                "from_type": from_obj.get("type"),
                "from_unit": from_obj.get("unit"),
                "from_property_id": from_obj.get("property_id"),
                "from_role": from_obj.get("role"),
                "from_company": from_obj.get("company"),
                "to": email.get("to"),
                "cc": email.get("cc"),
                "subject": email.get("subject"),
                "body": email.get("body"),
                "attachments": email.get("attachments"),
                "read": email.get("read"),
            }
        )

    df = pd.DataFrame(rows)

    for col in EMAIL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[EMAIL_COLUMNS]

    for col in [
        "id",
        "thread_id",
        "from_name",
        "from_email",
        "from_type",
        "from_unit",
        "from_property_id",
        "from_role",
        "from_company",
        "subject",
        "body",
    ]:
        df[col] = df[col].apply(_to_text)

    df["to"] = df["to"].apply(_to_list)
    df["cc"] = df["cc"].apply(_to_list)
    df["attachments"] = df["attachments"].apply(_to_list)
    df["read"] = df["read"].apply(_to_bool)

    df["thread_position"] = pd.to_numeric(df["thread_position"], errors="coerce")
    df["thread_position"] = df["thread_position"].fillna(10**9).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    missing_thread_mask = df["thread_id"] == ""
    if missing_thread_mask.any():
        df.loc[missing_thread_mask, "thread_id"] = df.loc[missing_thread_mask, "id"].apply(
            lambda value: f"missing_thread_{value}" if value else "missing_thread"
        )

    missing_sender_mask = df["from_type"] == ""
    df.loc[missing_sender_mask, "from_type"] = "unknown"

    empty_property_mask = df["from_property_id"] == ""
    df.loc[empty_property_mask, "from_property_id"] = None

    df = df.sort_values(
        by=["thread_id", "thread_position", "timestamp"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def load_properties(raw: dict) -> pd.DataFrame:
    properties = (raw.get("metadata") or {}).get("properties") or []

    rows: list[dict[str, Any]] = []
    if isinstance(properties, list):
        for prop in properties:
            if not isinstance(prop, dict):
                continue
            rows.append(
                {
                    "property_id": _to_text(prop.get("id")) or None,
                    "property_name": _to_text(prop.get("name")) or "Unknown Property",
                    "property_type": _to_text(prop.get("type")),
                    "property_units": prop.get("units"),
                    "property_manager": _to_text(prop.get("manager")),
                }
            )

    props_df = pd.DataFrame(rows)
    if props_df.empty:
        return pd.DataFrame(columns=PROPERTY_COLUMNS)

    props_df = props_df[PROPERTY_COLUMNS].drop_duplicates(subset=["property_id"], keep="first")
    return props_df


def infer_property_id_within_thread(emails_df: pd.DataFrame) -> pd.DataFrame:
    inferred = emails_df.copy()

    inferred["resolved_property_id"] = inferred["from_property_id"]

    for thread_id, group in inferred.groupby("thread_id", dropna=False):
        non_empty = [
            str(value)
            for value in group["from_property_id"].tolist()
            if value is not None and str(value).strip()
        ]
        selected = Counter(non_empty).most_common(1)[0][0] if non_empty else None

        if selected is not None:
            missing_mask = (inferred["thread_id"] == thread_id) & (
                inferred["resolved_property_id"].isna()
                | (inferred["resolved_property_id"].astype(str).str.strip() == "")
            )
            inferred.loc[missing_mask, "resolved_property_id"] = selected

    return inferred


def enrich_with_property_metadata(emails_df: pd.DataFrame, properties_df: pd.DataFrame) -> pd.DataFrame:
    if properties_df.empty:
        enriched = emails_df.copy()
        enriched["property_id"] = enriched["resolved_property_id"]
        enriched["property_name"] = "Unknown Property"
        enriched["property_type"] = ""
        enriched["property_units"] = None
        enriched["property_manager"] = ""
        return enriched

    enriched = emails_df.merge(
        properties_df,
        how="left",
        left_on="resolved_property_id",
        right_on="property_id",
        suffixes=("", "_meta"),
    )

    enriched["property_id"] = enriched["resolved_property_id"]
    missing_name = enriched["property_name"].isna() | (
        enriched["property_name"].astype(str).str.strip() == ""
    )
    enriched.loc[missing_name, "property_name"] = "Unknown Property"

    return enriched


def group_emails_by_thread(emails_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for thread_id, group in emails_df.groupby("thread_id", dropna=False):
        grouped[str(thread_id)] = group.sort_values(
            by=["thread_position", "timestamp"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return grouped


def load_and_prepare(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = load_json(dataset_path)
    emails_df = flatten_emails(raw)
    properties_df = load_properties(raw)
    emails_df = infer_property_id_within_thread(emails_df)
    emails_df = enrich_with_property_metadata(emails_df, properties_df)

    warnings: list[str] = []

    invalid_ts = int(emails_df["timestamp"].isna().sum())
    if invalid_ts:
        warnings.append(f"{invalid_ts} emails have invalid timestamps.")

    unresolved_prop = int(emails_df["resolved_property_id"].isna().sum())
    if unresolved_prop:
        warnings.append(
            f"{unresolved_prop} emails still have unknown property_id after thread inference."
        )

    unknown_sender = int((emails_df["from_type"] == "unknown").sum())
    if unknown_sender:
        warnings.append(f"{unknown_sender} emails have unknown from.type.")

    return emails_df, properties_df, sorted(set(warnings))
