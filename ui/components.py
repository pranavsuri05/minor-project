"""Small HTML/Streamlit building blocks that keep the pages consistent."""

from __future__ import annotations

import html

import streamlit as st

STATUS_LABELS = {
    "pending": "Pending",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "ok": "Completed",
    "warn": "Warning",
    "interrupted": "Interrupted",
}
STATUS_BADGE = {
    "pending": "pending",
    "running": "running",
    "completed": "ok",
    "failed": "fail",
    "interrupted": "warn",
}


def esc(text: object) -> str:
    return html.escape(str(text or ""))


def brand_header() -> None:
    st.markdown(
        '<div class="rp-brand"><div class="rp-brand-mark">RP</div>'
        '<div><div class="rp-brand-name">Research Platform</div>'
        '<div class="rp-brand-sub">Multi-agent literature review</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="rp-page-head"><h1>{esc(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="rp-section-label">{esc(text)}</div>', unsafe_allow_html=True)


def badge(label: str, tone: str) -> str:
    return f'<span class="rp-badge {tone}">{esc(label)}</span>'


def status_badge(status: str) -> str:
    return badge(STATUS_LABELS.get(status, status.title()), STATUS_BADGE.get(status, "pending"))


def metric(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f'<div class="rp-empty"><div class="rp-empty-title">{esc(title)}</div>'
        f'<div class="rp-empty-text">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str = "", meta: str = "") -> None:
    meta_html = f'<div class="rp-meta">{esc(meta)}</div>' if meta else ""
    st.markdown(
        f'<div class="rp-card"><div class="rp-card-title">{esc(title)}</div>{meta_html}{body_html}</div>',
        unsafe_allow_html=True,
    )


def stage_list_html(stages: list[tuple[str, str, str]], statuses: dict[str, str], messages: dict[str, str],
                    index_of: dict[str, int]) -> str:
    """Render the agent pipeline as a vertical list. Status comes from real stage events only."""
    rows = []
    for key, label, description in stages:
        status = statuses.get(key, "pending")
        marker = {"completed": "✓", "failed": "!", "running": "•"}.get(status, str(index_of[key] + 1))
        message = messages.get(key, "")
        msg_html = f'<div class="rp-stage-msg">{esc(message)}</div>' if message else ""
        rows.append(
            f'<div class="rp-stage {esc(status)}"><div class="rp-stage-dot">{esc(marker)}</div>'
            f'<div><div class="rp-stage-name">{esc(label)}</div>'
            f'<div class="rp-stage-desc">{esc(description)}</div>{msg_html}</div>'
            f'<div>{status_badge(status)}</div></div>'
        )
    return f'<div class="rp-card">{"".join(rows)}</div>'


def check_list_html(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        icon, color = ("✓", "var(--ok)") if check["passed"] else ("!", "var(--fail)")
        details = "".join(f'<div class="rp-meta">{esc(d)}</div>' for d in check.get("details", [])[:8])
        rows.append(
            f'<div class="rp-check"><div class="rp-check-icon" style="color:{color}">{icon}</div>'
            f'<div><div style="font-weight:550">{esc(check["label"])}</div>{details}</div></div>'
        )
    return f'<div class="rp-card">{"".join(rows)}</div>'
