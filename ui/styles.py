"""Theme and layout CSS. Colours are CSS variables, so Light, Dark and Auto only swap the token block."""

from __future__ import annotations

LIGHT = {
    "--bg": "#F5F4F0",
    "--sidebar": "#ECEBE6",
    "--card": "#FFFFFF",
    "--text": "#111113",
    "--muted": "#6B6E78",
    "--border": "#DEDBD3",
    "--accent": "#315EF5",
    "--accent-soft": "rgba(49, 94, 245, 0.10)",
    "--accent-text": "#FFFFFF",
    "--input": "#FFFFFF",
    "--hover": "rgba(17, 17, 19, 0.05)",
    "--ok": "#1F8A4C",
    "--ok-soft": "rgba(31, 138, 76, 0.12)",
    "--warn": "#A86A10",
    "--warn-soft": "rgba(168, 106, 16, 0.12)",
    "--fail": "#C2412D",
    "--fail-soft": "rgba(194, 65, 45, 0.12)",
    "--shadow": "0 1px 2px rgba(20, 20, 30, 0.04)",
    "--color-scheme": "light",
}

DARK = {
    "--bg": "#08090C",
    "--sidebar": "#101116",
    "--card": "#15171E",
    "--text": "#F5F5F7",
    "--muted": "#A0A3AD",
    "--border": "#292B34",
    "--accent": "#6485FF",
    "--accent-soft": "rgba(100, 133, 255, 0.16)",
    "--accent-text": "#08090C",
    "--input": "#1B1D25",
    "--hover": "rgba(245, 245, 247, 0.06)",
    "--ok": "#4CC38A",
    "--ok-soft": "rgba(76, 195, 138, 0.14)",
    "--warn": "#E7B25B",
    "--warn-soft": "rgba(231, 178, 91, 0.14)",
    "--fail": "#F0785F",
    "--fail-soft": "rgba(240, 120, 95, 0.14)",
    "--shadow": "none",
    "--color-scheme": "dark",
}

MODES = ("Light", "Dark", "Auto")


def _block(selector: str, tokens: dict[str, str]) -> str:
    body = "\n".join(f"  {k}: {v};" for k, v in tokens.items())
    return f"{selector} {{\n{body}\n}}"


def _tokens_css(mode: str) -> str:
    """Token blocks for the chosen appearance. Auto follows the operating system via prefers-color-scheme."""
    if mode == "Dark":
        return _block(":root", DARK)
    if mode == "Auto":
        return (
            _block(":root", LIGHT)
            + "\n@media (prefers-color-scheme: dark) {\n"
            + _block(":root:not([data-theme='light'])", DARK)
            + "\n}"
        )
    return _block(":root", LIGHT)


_STATIC_CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Inter, sans-serif;
  color-scheme: var(--color-scheme);
}
[data-testid="stHeader"] { background: transparent !important; }
/* Hide the Deploy button, menu and footer, but not the toolbar itself: it contains the button that re-opens a collapsed sidebar. */
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], #MainMenu, footer { visibility: hidden; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; color: var(--text); }
[data-testid="stExpandSidebarButton"] span, [data-testid="stSidebarCollapsedControl"] span { color: var(--text) !important; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }

[data-testid="stSidebar"] { background: var(--sidebar) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

h1, h2, h3, h4, h5, p, li, label, span, div, td, th { color: var(--text); }
h1 { font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.35rem !important; font-weight: 620 !important; letter-spacing: -0.01em; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
p, li, .stMarkdown { color: var(--text); line-height: 1.6; }
.rp-muted, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Inputs */
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px !important;
  color: var(--text) !important;
}
textarea, input { color: var(--text) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stFileUploader"] section {
  background: var(--card) !important; border: 1px dashed var(--border) !important; border-radius: 14px !important;
}
[data-testid="stFileUploader"] section * { color: var(--muted); }
[data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFile"] * { color: var(--text) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] { background: var(--card) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important; border: 1px solid var(--border) !important; background: var(--card) !important;
  color: var(--text) !important; font-weight: 550 !important; padding: 0.5rem 1.05rem !important;
  box-shadow: var(--shadow); transition: background 0.15s ease, border-color 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent) !important; background: var(--hover) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--accent) !important; color: var(--accent-text) !important; border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.06); background: var(--accent) !important; }
.stButton > button:disabled { opacity: 0.45; }

/* Radio as segmented control */
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.35rem; }
[data-testid="stRadio"] label p { font-size: 0.93rem; }

/* Tabs */
[data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
[data-baseweb="tab-highlight"] { background: var(--accent) !important; }

/* Containers and widgets */
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; box-shadow: var(--shadow); }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 640; }
[data-testid="stExpander"] { background: var(--card); border: 1px solid var(--border) !important; border-radius: 14px !important; }
[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 550; }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stProgress"] > div > div > div { background: var(--accent) !important; }
hr { border-color: var(--border) !important; }

/* Application components */
.rp-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.6rem; }
.rp-brand-mark { width: 30px; height: 30px; border-radius: 9px; background: var(--accent); color: var(--accent-text);
  display: grid; place-items: center; font-weight: 700; font-size: 0.95rem; }
.rp-brand-name { font-weight: 650; font-size: 0.98rem; line-height: 1.15; }
.rp-brand-sub { color: var(--muted); font-size: 0.74rem; }
.rp-page-head { margin-bottom: 1.6rem; }
.rp-page-head p { color: var(--muted); margin: 0.35rem 0 0 0; font-size: 0.98rem; }
.rp-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px;
  box-shadow: var(--shadow); margin-bottom: 0.9rem; }
.rp-card h4 { margin: 0 0 0.35rem 0; font-size: 1rem; font-weight: 600; }
.rp-card-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.2rem; }
.rp-meta { color: var(--muted); font-size: 0.85rem; }
.rp-section-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em;
  font-weight: 600; margin: 1.4rem 0 0.6rem 0; }
.rp-badge { display: inline-flex; align-items: center; gap: 0.35rem; padding: 3px 10px; border-radius: 999px;
  font-size: 0.76rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.rp-badge.ok { color: var(--ok); background: var(--ok-soft); }
.rp-badge.running { color: var(--accent); background: var(--accent-soft); }
.rp-badge.warn { color: var(--warn); background: var(--warn-soft); }
.rp-badge.fail { color: var(--fail); background: var(--fail-soft); }
.rp-badge.pending { color: var(--muted); background: var(--hover); border-color: var(--border); }
.rp-stage { display: grid; grid-template-columns: 34px 1fr auto; gap: 0.8rem; align-items: start;
  padding: 0.85rem 0; border-bottom: 1px solid var(--border); }
.rp-stage:last-child { border-bottom: none; }
.rp-stage-dot { width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-size: 0.8rem;
  font-weight: 700; background: var(--hover); color: var(--muted); border: 1px solid var(--border); }
.rp-stage.running .rp-stage-dot { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.rp-stage.completed .rp-stage-dot { color: var(--ok); border-color: var(--ok); background: var(--ok-soft); }
.rp-stage.failed .rp-stage-dot { color: var(--fail); border-color: var(--fail); background: var(--fail-soft); }
.rp-stage-name { font-weight: 600; font-size: 0.95rem; }
.rp-stage-desc { color: var(--muted); font-size: 0.84rem; margin-top: 0.1rem; }
.rp-stage-msg { color: var(--text); font-size: 0.84rem; margin-top: 0.3rem; opacity: 0.85; }
.rp-empty { text-align: center; padding: 2.4rem 1.2rem; border: 1px dashed var(--border); border-radius: 14px;
  background: var(--card); }
.rp-empty-title { font-weight: 600; margin-bottom: 0.3rem; }
.rp-empty-text { color: var(--muted); font-size: 0.92rem; }
.rp-score { font-size: 2.1rem; font-weight: 650; letter-spacing: -0.02em; }
.rp-check { display: flex; gap: 0.6rem; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.rp-check:last-child { border-bottom: none; }
.rp-check-icon { width: 22px; font-weight: 700; }
.rp-reference { font-size: 0.92rem; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.rp-pill { display: inline-block; font-size: 0.72rem; font-weight: 600; color: var(--muted); border: 1px solid var(--border);
  border-radius: 6px; padding: 1px 7px; margin-right: 0.35rem; }

@media (max-width: 760px) {
  .block-container { padding-left: 16px !important; padding-right: 16px !important; padding-top: 1.2rem; }
  h1 { font-size: 1.55rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
}
"""


def build_css(mode: str) -> str:
    if mode not in MODES:
        mode = "Light"
    return f"<style>\n{_tokens_css(mode)}\n{_STATIC_CSS}\n</style>"
