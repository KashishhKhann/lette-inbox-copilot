from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from llm import generate_thread_llm_fields
from parsing import load_and_prepare_data
from scoring import classify_issue, label_urgency, score_urgency


THREAD_COLUMNS = [
    "thread_id",
    "property_id",
    "property_name",
    "primary_sender_type",
    "latest_sender_type",
    "participants",
    "participant_types",
    "issue_type",
    "summary",
    "urgency_score",
    "urgency_label",
    "recommended_action",
    "reasoning",
    "unread_count",
    "attachment_count",
    "last_timestamp",
]


def _choose_thread_property(group: pd.DataFrame) -> tuple[str | None, str]:
    property_ids = [
        str(value)
        for value in group["from_property_id"].tolist()
        if value is not None and str(value).strip()
    ]

    if not property_ids:
        return None, "Unknown Property"

    chosen_property_id = Counter(property_ids).most_common(1)[0][0]
    matching = group[group["from_property_id"] == chosen_property_id]

    if not matching.empty:
        property_name = str(matching.iloc[0].get("property_name", "Unknown Property") or "Unknown Property")
    else:
        property_name = "Unknown Property"

    if property_name == "Unknown Property" and chosen_property_id:
        property_name = f"Unknown Property ({chosen_property_id})"

    return chosen_property_id, property_name


def _collect_participants(group: pd.DataFrame) -> list[str]:
    participants: set[str] = set()

    for _, row in group.iterrows():
        sender_email = str(row.get("from_email", "") or "").strip()
        if sender_email:
            participants.add(sender_email)

        for recipient in row.get("to", []):
            target = str(recipient).strip()
            if target:
                participants.add(target)

        for recipient in row.get("cc", []):
            target = str(recipient).strip()
            if target:
                participants.add(target)

    return sorted(participants)


def _collect_participant_types(group: pd.DataFrame) -> list[str]:
    types = sorted(
        {
            str(value).strip().lower()
            for value in group["from_type"].tolist()
            if str(value).strip()
        }
    )
    return types or ["unknown"]


def _build_message_dicts(group: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = group.sort_values(
        by=["thread_position", "timestamp"],
        ascending=[True, True],
        kind="mergesort",
    )

    messages: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        timestamp = row.get("timestamp")
        ts_text = timestamp.isoformat() if pd.notna(timestamp) else ""
        messages.append(
            {
                "id": row.get("id", ""),
                "thread_position": int(row.get("thread_position", 10**9) or 10**9),
                "timestamp": ts_text,
                "from_type": row.get("from_type", "unknown"),
                "from_email": row.get("from_email", ""),
                "subject": row.get("subject", ""),
                "body": row.get("body", ""),
                "read": bool(row.get("read", False)),
                "attachments": row.get("attachments", []),
            }
        )

    return messages


def build_thread_dataframe(emails_df: pd.DataFrame, llm_enabled: bool = True) -> tuple[pd.DataFrame, list[str]]:
    if emails_df.empty:
        return pd.DataFrame(columns=THREAD_COLUMNS), ["No emails available to aggregate."]

    warnings: list[str] = []
    records: list[dict[str, Any]] = []

    grouped = emails_df.groupby("thread_id", dropna=False)

    for thread_id, group in grouped:
        ordered = group.sort_values(
            by=["thread_position", "timestamp"],
            ascending=[True, True],
            kind="mergesort",
        )

        first_row = ordered.iloc[0]
        last_row = ordered.iloc[-1]

        property_id, property_name = _choose_thread_property(ordered)
        participants = _collect_participants(ordered)
        participant_types = _collect_participant_types(ordered)

        subject_concat = " | ".join(
            [str(value).strip() for value in ordered["subject"].tolist() if str(value).strip()]
        )
        body_concat = "\n\n".join(
            [str(value).strip() for value in ordered["body"].tolist() if str(value).strip()]
        )
        full_text = f"{subject_concat}\n{body_concat}".strip()

        issue_type = classify_issue(full_text)

        unread_count = int((~ordered["read"]).sum())
        attachment_count = int(ordered["attachments"].apply(len).sum())

        score, reasons = score_urgency(
            subject=subject_concat,
            body=body_concat,
            sender_type=str(last_row.get("from_type", "unknown") or "unknown"),
            unread=unread_count,
            attachment_count=attachment_count,
        )
        urgency_label = label_urgency(score)

        message_dicts = _build_message_dicts(ordered)
        thread_bundle = {
            "thread_id": str(thread_id),
            "property_id": property_id,
            "property_name": property_name,
            "primary_sender_type": str(first_row.get("from_type", "unknown") or "unknown"),
            "latest_sender_type": str(last_row.get("from_type", "unknown") or "unknown"),
            "participants": participants,
            "participant_types": participant_types,
            "issue_type": issue_type,
            "urgency_score": score,
            "urgency_label": urgency_label,
            "unread_count": unread_count,
            "attachment_count": attachment_count,
            "messages": message_dicts,
        }

        if llm_enabled:
            llm_fields = generate_thread_llm_fields(thread_bundle)
            if llm_fields.get("warning"):
                warnings.append(llm_fields["warning"])
            summary = llm_fields.get("summary", "")
            recommended_action = llm_fields.get("recommended_action", "")
        else:
            summary = (
                f"{property_name}: {issue_type.replace('_', ' ')} thread with "
                f"{unread_count} unread message(s)."
            )
            recommended_action = "Review latest message and assign next owner action."

        records.append(
            {
                "thread_id": str(thread_id),
                "property_id": property_id,
                "property_name": property_name,
                "primary_sender_type": str(first_row.get("from_type", "unknown") or "unknown"),
                "latest_sender_type": str(last_row.get("from_type", "unknown") or "unknown"),
                "participants": participants,
                "participant_types": participant_types,
                "issue_type": issue_type,
                "summary": summary,
                "urgency_score": score,
                "urgency_label": urgency_label,
                "recommended_action": recommended_action,
                "reasoning": "; ".join(reasons),
                "unread_count": unread_count,
                "attachment_count": attachment_count,
                "last_timestamp": ordered["timestamp"].max(),
            }
        )

    thread_df = pd.DataFrame(records)
    thread_df = thread_df.sort_values(
        by=["urgency_score", "last_timestamp"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    warnings = sorted(set(warnings))

    for column in THREAD_COLUMNS:
        if column not in thread_df.columns:
            thread_df[column] = None

    return thread_df[THREAD_COLUMNS], warnings


def run_pipeline(dataset_path: str, llm_enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    emails_df, _, parse_warnings = load_and_prepare_data(dataset_path)
    thread_df, thread_warnings = build_thread_dataframe(emails_df, llm_enabled=llm_enabled)

    mixed_property_threads = 0
    if not emails_df.empty:
        for _, group in emails_df.groupby("thread_id"):
            ids = {
                str(value)
                for value in group["from_property_id"].tolist()
                if value is not None and str(value).strip()
            }
            if len(ids) > 1:
                mixed_property_threads += 1

    extra_warnings: list[str] = []
    if mixed_property_threads:
        extra_warnings.append(
            f"{mixed_property_threads} threads had multiple property_id values; most frequent value was used."
        )

    warnings = sorted(set(parse_warnings + thread_warnings + extra_warnings))
    return thread_df, emails_df, warnings
