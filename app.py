from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import run_pipeline


DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", "data/proptech-test-data.json")


@st.cache_data(show_spinner=False)
def cached_pipeline(dataset_path: str, llm_enabled: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return run_pipeline(dataset_path=dataset_path, llm_enabled=llm_enabled)


def _format_list(values: list[str]) -> str:
    if not values:
        return "-"
    return ", ".join(values)


def _render_filters(threads_df: pd.DataFrame) -> dict[str, list[str]]:
    st.sidebar.header("Filters")

    urgency_values = sorted(threads_df["urgency_label"].dropna().unique().tolist())
    property_values = sorted(threads_df["property_name"].dropna().unique().tolist())
    sender_values = sorted(threads_df["latest_sender_type"].dropna().unique().tolist())
    issue_values = sorted(threads_df["issue_type"].dropna().unique().tolist())

    return {
        "urgency": st.sidebar.multiselect("Urgency", urgency_values, default=urgency_values),
        "property": st.sidebar.multiselect("Property", property_values, default=property_values),
        "sender": st.sidebar.multiselect(
            "Latest Sender Type", sender_values, default=sender_values
        ),
        "issue": st.sidebar.multiselect("Issue Type", issue_values, default=issue_values),
    }


def _apply_filters(threads_df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = threads_df.copy()

    if filters["urgency"]:
        filtered = filtered[filtered["urgency_label"].isin(filters["urgency"])]
    if filters["property"]:
        filtered = filtered[filtered["property_name"].isin(filters["property"])]
    if filters["sender"]:
        filtered = filtered[filtered["latest_sender_type"].isin(filters["sender"])]
    if filters["issue"]:
        filtered = filtered[filtered["issue_type"].isin(filters["issue"])]

    return filtered


def _render_metrics(threads_df: pd.DataFrame) -> None:
    total_threads = int(len(threads_df))
    critical_threads = int((threads_df["urgency_label"] == "critical").sum())
    unread_threads = int((threads_df["unread_count"] > 0).sum())
    prospects_awaiting_followup = int(
        ((threads_df["latest_sender_type"] == "prospect") & (threads_df["unread_count"] > 0)).sum()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Threads", total_threads)
    col2.metric("Critical Threads", critical_threads)
    col3.metric("Unread Threads", unread_threads)
    col4.metric("Prospects Awaiting Follow-up", prospects_awaiting_followup)


def _render_thread_detail(selected_thread: pd.Series, emails_df: pd.DataFrame) -> None:
    st.subheader("Thread Detail")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Summary")
        st.write(selected_thread.get("summary", ""))

        st.markdown("### Recommended Action")
        st.write(selected_thread.get("recommended_action", ""))

        st.markdown("### Urgency")
        st.write(
            f"Score: {selected_thread.get('urgency_score', 0)} "
            f"({selected_thread.get('urgency_label', 'low')})"
        )
        st.write(f"Reasons: {selected_thread.get('reasoning', '')}")

    with right:
        st.markdown("### Context")
        st.write(f"Thread ID: {selected_thread.get('thread_id', '')}")
        st.write(f"Property: {selected_thread.get('property_name', '')}")
        st.write(f"Property ID: {selected_thread.get('property_id', '-') or '-'}")
        st.write(f"Issue Type: {selected_thread.get('issue_type', '')}")
        st.write(f"Primary Sender Type: {selected_thread.get('primary_sender_type', '')}")
        st.write(f"Latest Sender Type: {selected_thread.get('latest_sender_type', '')}")
        st.write(f"Participant Types: {_format_list(selected_thread.get('participant_types', []))}")
        st.write(f"Participants: {_format_list(selected_thread.get('participants', []))}")
        st.write(f"Last Timestamp: {selected_thread.get('last_timestamp', '')}")

    st.markdown("### Message Timeline")
    thread_emails = emails_df[emails_df["thread_id"] == selected_thread["thread_id"]].copy()
    thread_emails = thread_emails.sort_values(
        by=["thread_position", "timestamp"],
        ascending=[True, True],
        kind="mergesort",
    )

    if thread_emails.empty:
        st.info("No emails available for this thread.")
        return

    for _, row in thread_emails.iterrows():
        ts = row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else "unknown_time"
        title = (
            f"#{int(row['thread_position'])} | {ts} | {row['from_type']} | "
            f"{row['subject'] or '(no subject)'}"
        )
        with st.expander(title):
            st.write(f"From: {row['from_name']} <{row['from_email']}>")
            st.write(f"To: {_format_list(row['to'])}")
            st.write(f"CC: {_format_list(row['cc'])}")
            st.write(f"Read: {row['read']}")
            st.write(f"Attachments: {_format_list(row['attachments'])}")
            st.write(row["body"] or "(empty body)")


def main() -> None:
    st.set_page_config(page_title="Lette Inbox Copilot", layout="wide")
    st.title("Lette Inbox Copilot")
    st.caption("Hackathon MVP: deterministic urgency + optional OpenAI summary/action")

    st.sidebar.header("Data")
    dataset_path = st.sidebar.text_input("Dataset Path", value=DEFAULT_DATASET_PATH)
    llm_enabled = st.sidebar.checkbox("Use OpenAI for summary/action", value=True)
    if st.sidebar.button("Re-run"):
        cached_pipeline.clear()

    if not Path(dataset_path).exists():
        st.error(f"Dataset not found: {dataset_path}")
        st.stop()

    try:
        with st.spinner("Building inbox triage view..."):
            threads_df, emails_df, warnings = cached_pipeline(dataset_path, llm_enabled)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        st.stop()

    if warnings:
        with st.expander("Warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)

    if threads_df.empty:
        st.info("No threads available.")
        st.stop()

    _render_metrics(threads_df)

    filters = _render_filters(threads_df)
    filtered_df = _apply_filters(threads_df, filters)

    st.subheader("Ranked Inbox")
    if filtered_df.empty:
        st.info("No matching threads for current filters.")
        st.stop()

    table_df = filtered_df[
        [
            "thread_id",
            "property_name",
            "issue_type",
            "latest_sender_type",
            "urgency_label",
            "urgency_score",
            "unread_count",
            "attachment_count",
            "last_timestamp",
            "summary",
        ]
    ].copy()

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    def _option_label(thread_id: str) -> str:
        row = filtered_df[filtered_df["thread_id"] == thread_id].iloc[0]
        return (
            f"{thread_id} | {row['urgency_label']} ({row['urgency_score']}) | "
            f"{row['property_name']} | {row['issue_type']}"
        )

    selected_thread_id = st.selectbox(
        "Select thread",
        options=filtered_df["thread_id"].tolist(),
        format_func=_option_label,
    )

    selected_thread = filtered_df[filtered_df["thread_id"] == selected_thread_id].iloc[0]
    _render_thread_detail(selected_thread, emails_df)


if __name__ == "__main__":
    main()
