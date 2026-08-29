"""Minimal LiteLLM JSON client for ScholarTrace."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Sequence

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        request_timeout: float = 180.0,
        max_retries: int = 1,
        temperature: float | None = None,
        allow_json_repair: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.temperature = default_temperature_for_model(model) if temperature is None else temperature
        self.allow_json_repair = allow_json_repair

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        image_urls: Sequence[str] | None = None,
        recorder: Any | None = None,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        kwargs = self._build_kwargs(
            prompt,
            system=system,
            temperature=temperature,
            image_urls=image_urls,
        )
        selected_temperature = self.temperature if temperature is None else temperature
        last_exc: Exception | None = None
        started_total = time.perf_counter()
        attempts = 0
        try:
            import litellm
        except Exception as exc:
            error = f"{type(exc).__name__}: {_redact_secret(str(exc), self.api_key)}"
            _record_call_error(
                recorder,
                call_id,
                error=error,
                attempt_count=attempts,
                latency_seconds=time.perf_counter() - started_total,
                model=self.model,
                temperature=selected_temperature,
            )
            raise RuntimeError(error) from None
        for attempt in range(self.max_retries + 1):
            attempts += 1
            started = time.perf_counter()
            try:
                response = litellm.completion(**kwargs)
            except Exception as exc:
                last_exc = exc
                elapsed = time.perf_counter() - started
                safe_error = _redact_secret(str(exc), self.api_key)
                log.warning(
                    "LLM call failed attempt=%s/%s elapsed=%.2fs model=%s error=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    elapsed,
                    self.model,
                    safe_error,
                )
            else:
                elapsed = time.perf_counter() - started
                log.info("LLM call succeeded attempt=%s elapsed=%.2fs", attempt + 1, elapsed)
                raw_response = extract_content(response)
                safe_raw_response = _redact_secret(raw_response, self.api_key)
                try:
                    if self.allow_json_repair:
                        parsed_response, json_repaired = parse_json_lenient_with_metadata(
                            safe_raw_response
                        )
                    else:
                        parsed_response = parse_json_strict(safe_raw_response)
                        json_repaired = False
                except Exception as exc:
                    error_prefix = "" if self.allow_json_repair else "invalid_json:"
                    safe_error = (
                        f"{error_prefix}{type(exc).__name__}: "
                        f"{_redact_secret(str(exc), self.api_key)}"
                    )
                    _record_call_completion(
                        recorder,
                        call_id,
                        raw_response=safe_raw_response,
                        parsed_response=None,
                        json_repaired=self.allow_json_repair,
                        error=safe_error,
                        attempt_count=attempts,
                        latency_seconds=time.perf_counter() - started_total,
                        response=response,
                        model=self.model,
                        temperature=selected_temperature,
                    )
                    if isinstance(exc, ValueError):
                        raise ValueError(safe_error) from None
                    raise RuntimeError(safe_error) from None
                _record_call_completion(
                    recorder,
                    call_id,
                    raw_response=safe_raw_response,
                    parsed_response=parsed_response,
                    json_repaired=json_repaired,
                    error=None,
                    attempt_count=attempts,
                    latency_seconds=time.perf_counter() - started_total,
                    response=response,
                    model=self.model,
                    temperature=selected_temperature,
                )
                return parsed_response
        error = (
            "LLM call failed after retries: "
            f"{type(last_exc).__name__ if last_exc is not None else 'UnknownError'}: "
            f"{_redact_secret(str(last_exc), self.api_key)}"
        )
        _record_call_error(
            recorder,
            call_id,
            error=error,
            attempt_count=attempts,
            latency_seconds=time.perf_counter() - started_total,
            model=self.model,
            temperature=selected_temperature,
        )
        raise RuntimeError(error) from None

    def _build_kwargs(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float | None,
        image_urls: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if image_urls:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                }
                for image_url in image_urls
            )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "request_timeout": self.request_timeout,
            "response_format": {"type": "json_object"},
            "max_retries": 0,
            "num_retries": 0,
        }
        selected_temperature = self.temperature if temperature is None else temperature
        if selected_temperature is not None:
            kwargs["temperature"] = selected_temperature
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs


def extract_content(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, KeyError):
        log.warning("Unexpected LLM response shape")
        return ""


def parse_json_lenient(raw: str) -> dict[str, Any]:
    return parse_json_lenient_with_metadata(raw)[0]


def parse_json_strict(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def parse_json_lenient_with_metadata(raw: str) -> tuple[dict[str, Any], bool]:
    raw = raw.strip()
    if not raw:
        return {}, False
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    json_repaired = False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        from json_repair import repair_json

        json_repaired = True
        data = json.loads(repair_json(raw))
    if not isinstance(data, dict):
        return {}, json_repaired
    return data, json_repaired


def _record_call_completion(
    recorder: Any | None,
    call_id: str | None,
    *,
    raw_response: str,
    parsed_response: dict[str, Any] | None,
    json_repaired: bool,
    error: str | None,
    attempt_count: int,
    latency_seconds: float,
    response: Any,
    model: str,
    temperature: float | None,
) -> None:
    if recorder is None or call_id is None:
        return
    update = getattr(recorder, "complete_call", None)
    if callable(update):
        update(
            call_id,
            raw_response=raw_response,
            parsed_response=parsed_response,
            json_repaired=json_repaired,
            error=error,
            attempt_count=attempt_count,
            latency_seconds=latency_seconds,
            prompt_tokens=_usage_value(response, "prompt_tokens"),
            completion_tokens=_usage_value(response, "completion_tokens"),
            total_tokens=_usage_value(response, "total_tokens"),
            model=model,
            temperature=temperature,
        )


def _record_call_error(
    recorder: Any | None,
    call_id: str | None,
    *,
    error: str,
    attempt_count: int,
    latency_seconds: float,
    model: str,
    temperature: float | None,
) -> None:
    if recorder is None or call_id is None:
        return
    update = getattr(recorder, "error_call", None)
    if callable(update):
        update(
            call_id,
            error=error,
            attempt_count=attempt_count,
            latency_seconds=latency_seconds,
            model=model,
            temperature=temperature,
        )


def _usage_value(response: Any, name: str) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def default_temperature_for_model(model: str) -> float | None:
    normalized = model.lower()
    if "gemini-3" in normalized:
        return None
    return 0.1
