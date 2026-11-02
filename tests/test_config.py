"""Configuration: Streamlit-secrets support, deployment defaults, and keeping keys out of logs."""

import config
from config import build_settings
from services.llm import LLMClient

KEY_VARS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL")


def _clean_env(monkeypatch):
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)


def test_deployment_defaults_are_gemini_native_flash_lite(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setattr(config, "_secret", lambda name: "")
    settings = build_settings()
    assert (settings.llm_provider, settings.llm_model) == ("gemini_native", "gemini-3.5-flash-lite")
    assert settings.llm_configured is False  # no key yet: the app must show the setup message, not crash
    assert "Streamlit Community Cloud" in settings.llm_setup_message and "GEMINI_API_KEY" in settings.llm_setup_message


def test_secrets_fill_in_when_environment_is_blank_and_environment_wins_when_set(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setattr(config, "_secret", lambda name: {"GEMINI_API_KEY": "from-secrets"}.get(name, ""))
    monkeypatch.setenv("GEMINI_API_KEY", "")  # a blank value, like an unfilled .env template line
    settings = build_settings()
    assert settings.gemini_api_key == "from-secrets" and settings.llm_configured is True

    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert build_settings().gemini_api_key == "from-env"


def test_missing_streamlit_secrets_do_not_raise():
    # Outside a Streamlit app with no secrets.toml, the lookup must quietly return an empty string.
    assert config._secret("A_SETTING_THAT_DOES_NOT_EXIST") == ""


def test_api_keys_never_appear_in_repr_or_error_text(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setattr(config, "_secret", lambda name: "")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-value-1234")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret-value-5678")
    settings = build_settings()
    assert "gemini-secret-value-1234" not in repr(settings)
    assert "openai-secret-value-5678" not in repr(settings)

    client = LLMClient(settings)
    scrubbed = client._sanitize("failed: key=gemini-secret-value-1234 and openai-secret-value-5678")
    assert "secret-value" not in scrubbed and scrubbed.count("<redacted>") == 2
