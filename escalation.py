from __future__ import annotations

import pandas as pd


ESCALATION_TERMS = {
    "rtb",
    "solicitor",
    "legal action",
    "compensation",
    "still not fixed",
    "third time",
    "again",
    "environmental health",
}

WELFARE_SMELL_TERMS = {
    "smell",
    "unpleasant smell",
    "strong smell",
    "odour",
    "odor",
}

WELFARE_ABSENCE_TERMS = {
    "haven't seen",
    "have not seen",
    "not seen",
    "in over a week",
    "no sign of",
}

WELFARE_CHECK_TERMS = {
    "post is piling up",
    "post piling up",
    "check on",
    "welfare check",
    "should someone check",
}

PRIOR_INTERVENTION_TERMS = {
    "someone came out",
    "contractor",
    "visited",
    "painted over",
    "dehumidifier",
    "inspected",
}

PRIOR_FAILURE_TERMS = {
    "came back",
    "back within",
    "reported again",
    "again",
    "nothing happened",
    "still not fixed",
    "no actual fix",
}

MANAGEMENT_SENDER_TYPES = {"internal", "landlord"}
CONTRACTOR_SENDER_TYPES = {"contractor", "vendor"}


def _contains_escalation_language(thread_df: pd.DataFrame) -> bool:
    full_text = "\n".join(
        [
            str(subject or "") + " " + str(body or "")
            for subject, body in zip(thread_df["subject"].tolist(), thread_df["body"].tolist())
        ]
    ).lower()
    return any(term in full_text for term in ESCALATION_TERMS)


def _thread_text(thread_df: pd.DataFrame) -> str:
    return "\n".join(
        [
            str(subject or "") + " " + str(body or "")
            for subject, body in zip(thread_df["subject"].tolist(), thread_df["body"].tolist())
        ]
    ).lower()


def _is_welfare_check_signal(text: str) -> bool:
    smell = any(term in text for term in WELFARE_SMELL_TERMS)
    absence = any(term in text for term in WELFARE_ABSENCE_TERMS)
    check = any(term in text for term in WELFARE_CHECK_TERMS)
    return smell and (absence or check)


def _mentions_prior_intervention_unresolved(text: str) -> bool:
    intervention = any(term in text for term in PRIOR_INTERVENTION_TERMS)
    unresolved = any(term in text for term in PRIOR_FAILURE_TERMS)
    return intervention and unresolved


def _has_sender_then_later_tenant(thread_df: pd.DataFrame, sender_types: set[str]) -> bool:
    sender_positions = [
        int(row.thread_position)
        for row in thread_df.itertuples()
        if str(row.from_type or "").strip().lower() in sender_types
    ]
    tenant_positions = [
        int(row.thread_position)
        for row in thread_df.itertuples()
        if str(row.from_type or "").strip().lower() == "tenant"
    ]

    if not sender_positions or not tenant_positions:
        return False

    return max(tenant_positions) > min(sender_positions)


def detect_human_required(thread_df: pd.DataFrame) -> dict:
    """Run human-required escalation rules. Returns first-class explainable result."""
    if thread_df.empty:
        return {
            "is_human": False,
            "handling_reason": "",
            "triggered_rule": "",
            "risk_flags": [],
        }

    ordered = thread_df.sort_values(
        by=["thread_position", "timestamp"],
        ascending=[True, True],
        kind="mergesort",
    )

    first_sender = str(ordered.iloc[0].get("from_type", "unknown") or "unknown").lower()
    latest_position = int(ordered.iloc[-1].get("thread_position", 1) or 1)
    full_text = _thread_text(ordered)

    triggered_rule = ""
    reason = ""
    risk_flags: list[str] = []

    if len(ordered) >= 3 and latest_position >= 3 and first_sender == "tenant":
        triggered_rule = "tenant_multi_touch"
        reason = "Thread reached 3+ touches and originated from tenant."
        risk_flags.append("repeat_unresolved")

    if not triggered_rule and _has_sender_then_later_tenant(ordered, CONTRACTOR_SENDER_TYPES):
        triggered_rule = "post_contractor_unresolved"
        reason = "Contractor responded and tenant emailed again afterward."
        risk_flags.append("post_contractor_unresolved")

    if not triggered_rule and _has_sender_then_later_tenant(ordered, MANAGEMENT_SENDER_TYPES):
        triggered_rule = "post_management_unresolved"
        reason = "Management responded and tenant emailed again afterward."
        risk_flags.append("post_management_unresolved")

    if not triggered_rule and first_sender == "tenant" and _mentions_prior_intervention_unresolved(full_text):
        triggered_rule = "historical_unresolved_after_intervention"
        reason = "Tenant describes prior intervention with issue still unresolved."
        risk_flags.append("post_contractor_unresolved")
        risk_flags.append("repeat_unresolved")

    if not triggered_rule and _is_welfare_check_signal(full_text):
        triggered_rule = "welfare_check_signal"
        reason = "Potential welfare-check signal detected from resident report."
        risk_flags.append("welfare_check")
        risk_flags.append("health_safety")

    if not triggered_rule and len(ordered) > 1 and _contains_escalation_language(ordered):
        triggered_rule = "escalation_language"
        reason = "Escalation/legal language detected in multi-email thread."
        risk_flags.append("legal_risk")

    return {
        "is_human": bool(triggered_rule),
        "handling_reason": reason,
        "triggered_rule": triggered_rule,
        "risk_flags": sorted(set(risk_flags)),
    }
