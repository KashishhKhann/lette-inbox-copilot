from __future__ import annotations


ISSUE_TYPES = {
    "emergency_maintenance",
    "maintenance",
    "complaint",
    "financial",
    "leasing",
    "move_out",
    "legal",
    "operational_internal",
    "vendor_management",
}


EMERGENCY_SIGNALS = {
    "leak",
    "water leak",
    "electrical hazard",
    "electrical issue",
    "no heating",
    "no heat",
    "no hot water",
    "fire alarm",
    "mould",
    "mold",
    "damp",
    "health and safety",
    "baby",
    "elderly",
    "urgent",
    "emergency",
}

LEGAL_SIGNALS = {
    "rtb",
    "legal",
    "dispute",
    "tribunal",
    "solicitor",
    "court",
    "eviction",
    "evict",
}

FINANCIAL_SIGNALS = {
    "rent",
    "arrears",
    "invoice",
    "overdue invoice",
    "payment hold",
    "deposit",
    "refund",
    "payment",
}

LEASING_SIGNALS = {
    "viewing request",
    "viewing",
    "corporate let",
    "corporate let inquiry",
    "lease",
    "application",
    "prospect",
}

MOVE_OUT_SIGNALS = {
    "move out",
    "move-out",
    "vacate",
    "checkout",
    "check-out",
    "handover",
}

VENDOR_SIGNALS = {
    "contractor",
    "vendor",
    "quote",
    "work order",
    "dispatch",
    "sla",
}

COMPLAINT_SIGNALS = {
    "complaint",
    "noise",
    "harassment",
    "unhappy",
    "frustrated",
}

MAINTENANCE_SIGNALS = {
    "repair",
    "maintenance",
    "broken",
    "heating",
    "plumbing",
    "electrical",
    "appliance",
    "inspection",
}

REPEATED_FOLLOW_UP_SIGNALS = {
    "following up",
    "follow up again",
    "as mentioned",
    "still waiting",
    "still unresolved",
    "third email",
    "second reminder",
}

ISSUE_BASE_SCORE = {
    "emergency_maintenance": 65,
    "legal": 55,
    "financial": 45,
    "complaint": 40,
    "maintenance": 38,
    "move_out": 32,
    "vendor_management": 30,
    "leasing": 22,
    "operational_internal": 18,
}


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def classify_issue(text: str) -> str:
    """Classify issue type using transparent keyword rules."""
    content = (text or "").lower()

    if _contains_any(content, EMERGENCY_SIGNALS):
        return "emergency_maintenance"
    if _contains_any(content, LEGAL_SIGNALS):
        return "legal"
    if _contains_any(content, FINANCIAL_SIGNALS):
        return "financial"
    if _contains_any(content, LEASING_SIGNALS):
        return "leasing"
    if _contains_any(content, MOVE_OUT_SIGNALS):
        return "move_out"
    if _contains_any(content, VENDOR_SIGNALS):
        return "vendor_management"
    if _contains_any(content, COMPLAINT_SIGNALS):
        return "complaint"
    if _contains_any(content, MAINTENANCE_SIGNALS):
        return "maintenance"

    return "operational_internal"


def score_urgency(
    subject: str,
    body: str,
    sender_type: str,
    unread: int,
    attachment_count: int,
) -> tuple[int, list[str]]:
    """Return deterministic urgency score and fired rule reasons."""
    subject_text = (subject or "").lower()
    body_text = (body or "").lower()
    text = f"{subject_text}\n{body_text}"

    issue_type = classify_issue(text)
    score = ISSUE_BASE_SCORE[issue_type]
    reasons: list[str] = [f"issue_type={issue_type} base={score}"]

    if _contains_any(text, EMERGENCY_SIGNALS):
        score += 20
        reasons.append("emergency signal detected (+20)")

    if _contains_any(text, {"baby", "elderly", "health and safety"}):
        score += 10
        reasons.append("vulnerable resident / safety context (+10)")

    if _contains_any(text, LEGAL_SIGNALS):
        score += 14
        reasons.append("legal/RTB/dispute signal (+14)")

    if _contains_any(text, {"overdue invoice", "payment hold"}):
        score += 12
        reasons.append("contractor payment hold / overdue invoice (+12)")

    if _contains_any(text, REPEATED_FOLLOW_UP_SIGNALS):
        score += 10
        reasons.append("repeated unresolved follow-up signal (+10)")

    if _contains_any(text, {"viewing request", "corporate let inquiry"}):
        score -= 8
        reasons.append("commercial inquiry (non-emergency) (-8)")

    sender = (sender_type or "unknown").strip().lower()
    if sender == "legal":
        score += 10
        reasons.append("latest sender type legal (+10)")
    elif sender == "tenant":
        score += 6
        reasons.append("latest sender type tenant (+6)")
    elif sender == "prospect":
        score += 4
        reasons.append("latest sender type prospect (+4)")
    elif sender == "system":
        score -= 5
        reasons.append("latest sender type system (-5)")

    unread_points = min(max(int(unread), 0) * 5, 20)
    if unread_points:
        score += unread_points
        reasons.append(f"unread emails={unread} (+{unread_points})")

    attachment_points = min(max(int(attachment_count), 0) * 2, 8)
    if attachment_points:
        score += attachment_points
        reasons.append(f"attachments={attachment_count} (+{attachment_points})")

    score = max(0, min(100, int(round(score))))
    return score, reasons


def label_urgency(score: int) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"
