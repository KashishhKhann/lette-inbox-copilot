from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_EMAIL_COLUMNS = [
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


def load_json(path: str) -> dict:
    """Load and validate the top-level dataset object."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in dataset: {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Dataset must be a JSON object.")

    if not isinstance(raw.get("emails"), list):
        raise ValueError("Dataset must include an 'emails' list.")

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


def emails_to_dataframe(raw: dict) -> pd.DataFrame:
    """Flatten nested email objects into a row-wise dataframe."""
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

    emails_df = pd.DataFrame(rows)

    for column in REQUIRED_EMAIL_COLUMNS:
        if column not in emails_df.columns:
            emails_df[column] = None

    emails_df = emails_df[REQUIRED_EMAIL_COLUMNS]

    for column in [
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
        emails_df[column] = emails_df[column].apply(_to_text)

    emails_df["to"] = emails_df["to"].apply(_to_list)
    emails_df["cc"] = emails_df["cc"].apply(_to_list)
    emails_df["attachments"] = emails_df["attachments"].apply(_to_list)
    emails_df["read"] = emails_df["read"].apply(_to_bool)

    emails_df["thread_position"] = pd.to_numeric(emails_df["thread_position"], errors="coerce")
    emails_df["thread_position"] = emails_df["thread_position"].fillna(10**9).astype(int)
    emails_df["timestamp"] = pd.to_datetime(emails_df["timestamp"], errors="coerce", utc=True)

    missing_thread_mask = emails_df["thread_id"] == ""
    if missing_thread_mask.any():
        emails_df.loc[missing_thread_mask, "thread_id"] = emails_df.loc[
            missing_thread_mask, "id"
        ].apply(lambda value: f"missing_thread_{value}" if value else "missing_thread")

    missing_sender_type = emails_df["from_type"] == ""
    emails_df.loc[missing_sender_type, "from_type"] = "unknown"

    empty_property_id = emails_df["from_property_id"] == ""
    emails_df.loc[empty_property_id, "from_property_id"] = None

    # Required chronology rule: thread_position first, then timestamp.
    emails_df = emails_df.sort_values(
        by=["thread_id", "thread_position", "timestamp"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return emails_df


def properties_to_dataframe(raw: dict) -> pd.DataFrame:
    """Load properties metadata into a dataframe."""
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

    properties_df = pd.DataFrame(rows)
    if properties_df.empty:
        return pd.DataFrame(
            columns=[
                "property_id",
                "property_name",
                "property_type",
                "property_units",
                "property_manager",
            ]
        )

    properties_df = properties_df.drop_duplicates(subset=["property_id"], keep="first")
    return properties_df


def enrich_emails_with_properties(
    emails_df: pd.DataFrame,
    properties_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join property metadata onto each email row by from_property_id."""
    if emails_df.empty:
        return emails_df.copy()

    if properties_df.empty:
        enriched = emails_df.copy()
        enriched["property_id"] = enriched["from_property_id"]
        enriched["property_name"] = "Unknown Property"
        enriched["property_type"] = ""
        enriched["property_units"] = None
        enriched["property_manager"] = ""
        return enriched

    enriched = emails_df.merge(
        properties_df,
        how="left",
        left_on="from_property_id",
        right_on="property_id",
        suffixes=("", "_meta"),
    )

    enriched["property_id"] = enriched["from_property_id"]
    missing_name = enriched["property_name"].isna() | (enriched["property_name"].astype(str).str.strip() == "")
    enriched.loc[missing_name, "property_name"] = "Unknown Property"

    return enriched


def validate_email_dataframe(emails_df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []

    if emails_df.empty:
        warnings.append("No emails found in dataset.")
        return warnings

    invalid_ts = int(emails_df["timestamp"].isna().sum())
    if invalid_ts:
        warnings.append(f"{invalid_ts} emails have invalid timestamps.")

    missing_prop = int(emails_df["from_property_id"].isna().sum())
    if missing_prop:
        warnings.append(
            f"{missing_prop} emails are missing from.property_id; thread-level inference will be used."
        )

    unknown_sender_type = int((emails_df["from_type"] == "unknown").sum())
    if unknown_sender_type:
        warnings.append(
            f"{unknown_sender_type} emails are missing from.type and were set to 'unknown'."
        )

    return warnings


def load_and_prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = load_json(path)
    emails_df = emails_to_dataframe(raw)
    properties_df = properties_to_dataframe(raw)
    emails_df = enrich_emails_with_properties(emails_df, properties_df)
    warnings = validate_email_dataframe(emails_df)
    return emails_df, properties_df, warnings
