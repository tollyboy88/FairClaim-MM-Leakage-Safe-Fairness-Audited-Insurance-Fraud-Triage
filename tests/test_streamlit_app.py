from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_loads():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "FairClaim-MM decision support"


def test_all_dashboard_pages_render():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    expected = {
        "Single claim": "Assess one claim",
        "Batch upload": "Assess multiple claims",
        "Fairness & explanations": "Fairness and explanation audit",
        "Local history": "Local decision history",
    }
    for page, title in expected.items():
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert app.title[0].value == title
