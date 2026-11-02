"""The single LLM client shared by every agent.

Both supported providers (OpenAI and Gemini via its OpenAI-compatible endpoint) are reached
through LangChain's ChatOpenAI. Agents depend only on the `complete` / `complete_json`
interface, so tests can inject a deterministic fake without any network access.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for LLM failures that should be shown to the user."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model answered, but the answer could not be used (for example, invalid JSON)."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_text(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and leading/trailing prose."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if not starts:
        raise LLMOutputError("The model reply did not contain JSON.")
    start = min(starts)
    closer = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(closer)
    if end <= start:
        raise LLMOutputError("The model reply contained an incomplete JSON object.")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"The model reply contained invalid JSON: {exc}") from exc


class LLMClient:
    """One instance is created per process and passed to every agent."""

    JSON_INSTRUCTION = (
        "\n\nRespond with a single valid JSON value only. Do not wrap it in Markdown fences "
        "and do not add commentary."
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chat = None
        self._native = None  # google-genai client, used only for LLM_PROVIDER=gemini_native

    @property
    def is_configured(self) -> bool:
        return self.settings.llm_configured

    @property
    def model_label(self) -> str:
        return f"{self.settings.llm_provider}:{self.settings.llm_model}"

    def _client(self):
        if not self.is_configured:
            raise LLMNotConfiguredError(self.settings.llm_setup_message)
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.active_api_key,
                base_url=self.settings.llm_base_url,
                timeout=self.settings.llm_timeout_s,
                max_retries=self.settings.llm_max_retries,  # retries 429/5xx/timeouts with backoff
                max_tokens=self.settings.llm_max_tokens,
                temperature=0.2,
            )
        return self._chat

    def _sanitize(self, message: str) -> str:
        """Remove the API key from any text that may reach the UI or the log."""
        for key in {self.settings.gemini_api_key, self.settings.openai_api_key}:
            if key:
                message = message.replace(key, "<redacted>")
        return message

    def _complete_native(self, task: str, system: str, user: str) -> str:
        """Gemini through Google's official google-genai SDK (LLM_PROVIDER=gemini_native)."""
        from google import genai
        from google.genai import errors, types

        if not self.is_configured:
            raise LLMNotConfiguredError(self.settings.llm_setup_message)
        if self._native is None:
            self._native = genai.Client(api_key=self.settings.gemini_api_key)
        logger.info("LLM call start task=%s model=%s", task, self.model_label)
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                response = self._native.models.generate_content(
                    model=self.settings.llm_model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=self.settings.llm_max_tokens,
                        temperature=0.2,
                        http_options=types.HttpOptions(timeout=self.settings.llm_timeout_s * 1000),  # SDK uses ms
                    ),
                )
                break
            except errors.APIError as exc:
                transient = exc.code in (429, 500, 503, 504)
                if transient and attempt < self.settings.llm_max_retries:
                    delay = 2 ** (attempt + 1)  # 2s, then 4s
                    logger.warning("Gemini HTTP %s during task=%s; retrying in %ss", exc.code, task, delay)
                    time.sleep(delay)
                    continue
                raise LLMError(self._sanitize(
                    f"Gemini API returned HTTP {exc.code} during '{task}': {exc.message}")) from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Native Gemini call failed task=%s", task)
                raise LLMError(self._sanitize(f"Native Gemini call failed during '{task}': {exc}")) from exc
        text = response.text or ""
        logger.info("LLM call done task=%s chars=%d", task, len(text))
        return text

    def complete(self, task: str, system: str, user: str) -> str:
        """Return the model's text reply. `task` is a label used for logging."""
        if self.settings.llm_provider == "gemini_native":
            return self._complete_native(task, system, user)

        from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

        chat = self._client()
        logger.info("LLM call start task=%s model=%s", task, self.model_label)
        try:
            reply = chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except AuthenticationError as exc:
            raise LLMError(
                f"The {self.settings.llm_provider} API key was rejected (invalid or revoked). "
                "Check the key in your .env file."
            ) from exc
        except APITimeoutError as exc:
            raise LLMError(
                f"The LLM request timed out after {self.settings.llm_timeout_s}s (task: {task}). "
                "Try again or increase LLM_TIMEOUT_S."
            ) from exc
        except RateLimitError as exc:
            raise LLMError("The LLM provider rate limit or quota was reached. Try again later.") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not reach the LLM provider. Check your network connection.") from exc
        except Exception as exc:  # noqa: BLE001 - surface any provider failure with context
            logger.exception("LLM call failed task=%s", task)
            raise LLMError(self._sanitize(f"LLM call failed during '{task}': {exc}")) from exc
        text = reply.content if isinstance(reply.content, str) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in reply.content
        )
        logger.info("LLM call done task=%s chars=%d", task, len(text))
        return text

    def complete_json(self, task: str, system: str, user: str) -> Any:
        """Return parsed JSON. Retries once, asking the model to repair its own malformed output."""
        raw = self.complete(task, system + self.JSON_INSTRUCTION, user)
        try:
            return parse_json_text(raw)
        except LLMOutputError as first_error:
            logger.warning("Invalid JSON from task=%s; requesting one repair pass (%s)", task, first_error)
            repair_user = (
                "The previous answer was not valid JSON. Rewrite it as valid JSON that follows the "
                f"requested schema.\n\nPrevious answer:\n{raw[:6000]}"
            )
            repaired = self.complete(task + "_repair", system + self.JSON_INSTRUCTION, repair_user)
            return parse_json_text(repaired)
