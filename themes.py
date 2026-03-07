from __future__ import annotations

from collections import defaultdict

import pandas as pd

from llm import summarize_theme


def _risk_signature(risk_flags: list[str] | None) -> str:
    if not risk_flags:
        return "none"
    return "|".join(sorted(set([str(flag) for flag in risk_flags if str(flag).strip()])))


def _theme_label(issue_type: str, risk_signature: str) -> str:
    issue = (issue_type or "").lower()
    risk = (risk_signature or "").lower()

    if issue == "emergency_maintenance":
        return "urgent maintenance cluster"
    if issue == "legal" or "legal" in risk:
        return "legal/compliance risk"
    if issue == "financial":
        return "financial exposure"
    if issue in {"move_out", "leasing"}:
        return "tenancy turnover"
    if issue == "prospect":
        return "commercial opportunity"
    return "operational pressure"


def _severity_from_urgency(urgency_labels: list[str]) -> str:
    if "critical" in urgency_labels:
        return "critical"
    if "high" in urgency_labels:
        return "high"
    if "medium" in urgency_labels:
        return "medium"
    return "low"


def _fallback_theme_text(theme_label: str, thread_count: int, properties: list[str]) -> tuple[str, str]:
    insight = (
        f"{thread_count} related thread(s) indicate a recurring pattern: {theme_label}. "
        f"Affected properties: {', '.join(properties)}."
    )
    action = "Assign one portfolio owner, set SLA checkpoints, and track closure per property."
    return insight, action


def build_themes(
    threads_df: pd.DataFrame,
    llm_enabled: bool = True,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    if threads_df.empty:
        return pd.DataFrame(
            columns=[
                "theme_label",
                "severity",
                "affected_properties",
                "thread_count",
                "insight",
                "portfolio_action",
                "thread_ids",
            ]
        )

    clusters: dict[tuple, list[dict]] = defaultdict(list)

    for row in threads_df.to_dict(orient="records"):
        ts = row.get("latest_timestamp")
        if pd.isna(ts):
            day_bucket = "unknown_day"
        else:
            day_bucket = pd.Timestamp(ts).floor("D").isoformat()

        risk_signature = _risk_signature(row.get("risk_flags", []))
        key = (
            row.get("issue_type", "operational_internal"),
            row.get("urgency_label", "low"),
            row.get("property_name", "Unknown Property"),
            risk_signature,
            day_bucket,
        )
        clusters[key].append(row)

    theme_rows: list[dict] = []

    for key, rows in clusters.items():
        if len(rows) < min_cluster_size:
            continue

        issue_type, _urgency, _property_name, risk_signature, _day_bucket = key

        thread_ids = [str(row.get("thread_id", "")) for row in rows]
        affected_properties = sorted(
            set([str(row.get("property_name", "Unknown Property")) for row in rows])
        )
        urgency_labels = [str(row.get("urgency_label", "low")) for row in rows]

        theme_label = _theme_label(issue_type, risk_signature)
        severity = _severity_from_urgency(urgency_labels)
        fallback_insight, fallback_action = _fallback_theme_text(
            theme_label=theme_label,
            thread_count=len(rows),
            properties=affected_properties,
        )

        theme_obj = {
            "theme_label": theme_label,
            "severity": severity,
            "affected_properties": affected_properties,
            "thread_count": len(rows),
            "thread_ids": thread_ids,
            "insight": fallback_insight,
            "portfolio_action": fallback_action,
        }

        if llm_enabled:
            llm_text = summarize_theme(theme_obj)
            if isinstance(llm_text, dict):
                theme_obj["insight"] = llm_text.get("insight", theme_obj["insight"])
                theme_obj["portfolio_action"] = llm_text.get(
                    "portfolio_action", theme_obj["portfolio_action"]
                )

        theme_rows.append(theme_obj)

    if not theme_rows:
        return pd.DataFrame(
            columns=[
                "theme_label",
                "severity",
                "affected_properties",
                "thread_count",
                "insight",
                "portfolio_action",
                "thread_ids",
            ]
        )

    out_df = pd.DataFrame(theme_rows)
    out_df = out_df.sort_values(by=["thread_count", "severity"], ascending=[False, True]).reset_index(
        drop=True
    )
    return out_df
