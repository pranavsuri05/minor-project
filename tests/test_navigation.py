"""Navigation regression test: runs the real app and clicks real widgets (no LLM or network calls)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _page_title(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "rp-page-head" in m.value)


def test_overview_button_navigates_to_new_research_and_back():
    at = _run()
    assert at.session_state["nav"] == "Overview"

    # This click used to raise StreamlitWidgetAlreadyInstantiatedError.
    at.button(key="overview_new").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["nav"] == "New Research"
    assert "New research" in _page_title(at)

    at.sidebar.radio(key="nav").set_value("Overview").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["nav"] == "Overview"
    assert "Overview" in _page_title(at)

    # And the button still works a second time (Overview -> New Research again).
    at.button(key="overview_new").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["nav"] == "New Research"


def test_sidebar_navigation_reaches_every_page():
    at = _run()
    for label, heading in [
        ("Research History", "Research history"),
        ("Saved Papers", "Saved papers"),
        ("Settings", "Settings"),
        ("New Research", "New research"),
        ("Overview", "Overview"),
    ]:
        at.sidebar.radio(key="nav").set_value(label).run()
        assert not at.exception, (label, [e.value for e in at.exception])
        assert heading in _page_title(at), label
