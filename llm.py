from __future__ import annotations

import json
import os
from typing import Any

import requests


def _fallback_summary(thread_bundle: dict) -> str:
    property_name = thread_bundle.get("property_name", "Unknown Property")
    issue_type = thread_bundle.get("issue_type", "operational_internal").replace("_", " ")
    latest_sender = thread_bundle.get("latest_sender_type", "unknown")
    unread_count = int(thread_bundle.get("unread_count", 0) or 0)
    return (
        f"{property_name}: {issue_type}. Latest sender type is {latest_sender}. "
        f"Unread messages in thread: {unread_count}."
    )


def _fallback_action(thread_bundle: dict) -> str:
    issue_type = thread_bundle.get("issue_type", "operational_internal")
    urgency_label = thread_bundle.get("urgency_label", "medium")

    if issue_type == "emergency_maintenance":
        return (
            "Contact emergency contractor immediately, acknowledge resident, and confirm ETA/update."
        )
    if issue_type == "legal":
        return "Escalate to legal/compliance owner today and prepare the required response pack."
    if issue_type == "financial":
        return (
            "Validate account/invoice status, coordinate with finance, and send a clear payment update."
        )
    if issue_type == "leasing":
        return "Send availability and next booking step; assign follow-up owner and response deadline."
    if urgency_label in {"critical", "high"}:
        return "Acknowledge sender now and assign immediate owner action with deadline."

    return "Triage in the next work cycle and send a concise status update to participants."


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = text.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None

    return None


def _call_openai_thread_fields(thread_bundle: dict) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    timeout_s = int(os.getenv("OPENAI_TIMEOUT_S", "20"))

    snippet_lines = []
    for msg in thread_bundle.get("messages", [])[:6]:
        ts = msg.get("timestamp", "")
        sender = msg.get("from_type", "unknown")
        subject = msg.get("subject", "")
        body = (msg.get("body", "") or "").replace("\n", " ")
        if len(body) > 180:
            body = f"{body[:180]}..."
        snippet_lines.append(f"[{ts}] {sender} | {subject} | {body}")

    snippets = "\n".join(f"- {line}" for line in snippet_lines)

    system_msg = (
        "You assist a property manager. Be concise and practical. "
        "Return valid JSON only."
    )
    user_msg = (
        "Generate thread summary and next action for inbox triage.\n"
        "Output format: {\"summary\": string, \"recommended_action\": string}.\n"
        "Keep summary <= 45 words and action <= 30 words.\n\n"
        f"thread_id: {thread_bundle.get('thread_id', '')}\n"
        f"property_name: {thread_bundle.get('property_name', '')}\n"
        f"issue_type: {thread_bundle.get('issue_type', '')}\n"
        f"urgency_label: {thread_bundle.get('urgency_label', '')}\n"
        f"latest_sender_type: {thread_bundle.get('latest_sender_type', '')}\n"
        f"unread_count: {thread_bundle.get('unread_count', 0)}\n"
        f"attachment_count: {thread_bundle.get('attachment_count', 0)}\n"
        "messages:\n"
        f"{snippets}"
    )

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        },
        timeout=timeout_s,
    )
    response.raise_for_status()

    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    parsed = _extract_json(str(content))
    if not parsed:
        raise ValueError("OpenAI output was not valid JSON.")

    summary = str(parsed.get("summary", "")).strip()
    action = str(parsed.get("recommended_action", "")).strip()
    if not summary or not action:
        raise ValueError("OpenAI output missing summary or recommended_action.")

    return {"summary": summary, "recommended_action": action}


def generate_thread_llm_fields(thread_bundle: dict) -> dict[str, str]:
    """Generate summary/action, with deterministic fallback when key/call is unavailable."""
    try:
        llm_data = _call_openai_thread_fields(thread_bundle)
        return {
            "summary": llm_data["summary"],
            "recommended_action": llm_data["recommended_action"],
            "warning": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "summary": _fallback_summary(thread_bundle),
            "recommended_action": _fallback_action(thread_bundle),
            "warning": f"LLM fallback used: {type(exc).__name__}",
        }


def summarize_thread(thread_bundle: dict) -> str:
    return generate_thread_llm_fields(thread_bundle)["summary"]


def recommend_action(thread_bundle: dict) -> str:
    return generate_thread_llm_fields(thread_bundle)["recommended_action"]
