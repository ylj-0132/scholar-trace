"""Minimal LiteLLM JSON client for ScholarTrace."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Sequence

log = logging.getLogger(__name__)

PROMPT_LAYOUT_VERSIONS = {
    "standard": "field-order-v1",
    "cache-friendly": "explicit-breakpoints-v3",
}


def cache_friendly_prompt_parts(prompt: str) -> tuple[str, str]:
    """Return a reusable JSON prefix and dynamic suffix; concatenation is valid JSON."""
    payload = json.loads(prompt)
    if not isinstance(payload, dict):
        return "", prompt
    stable_fields = (
        "mechanism_audit_principles", "decision_checklist", "output_contract",
        "required_json_shape", "instructions", "investigation_target", "paper",
        "compact_page_index", "available_pages",
    )
    prefix = {key: payload[key] for key in stable_fields if key in payload}
    ordered = json.dumps({**prefix, **payload}, ensure_ascii=False, indent=2)
    if not prefix:
        return "", ordered
    if len(prefix) == len(payload):
        return ordered, ""
    # Exclude the closing newline/brace; the comma and dynamic fields stay outside
    # the breakpoint. Do not search field names inside arbitrary evidence text.
    boundary = len(json.dumps(prefix, ensure_ascii=False, indent=2)) - 2
    return ordered[:boundary], ordered[boundary:]


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
            cache_friendly=getattr(recorder, "prompt_layout", "standard") == "cache-friendly",
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
        cache_friendly: bool = False,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if cache_friendly or image_urls:
            content: list[dict[str, Any]] = []
            if cache_friendly:
                prefix, suffix = cache_friendly_prompt_parts(prompt)
                if system:
                    messages[0]["content"] = [{
                        "type": "text", "text": system,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }]
                if prefix:
                    content.append({
                        "type": "text", "text": prefix,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    })
                if suffix:
                    content.append({"type": "text", "text": suffix})
            else:
                content.append({"type": "text", "text": prompt})
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                }
                for image_url in (image_urls or ())
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
        if cache_friendly:
            # Pass through to the OpenAI-compatible HTTP body, including when
            # LiteLLM's model parameter allowlist predates explicit caching.
            kwargs["extra_body"] = {"prompt_cache_options": {"mode": "explicit"}}
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
            cached_prompt_tokens=_usage_value(response, "prompt_tokens_details", "cached_tokens"),
            cache_write_prompt_tokens=_usage_value(response, "prompt_tokens_details", "cache_write_tokens"),
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


def _usage_value(response: Any, *path: str) -> int | None:
    value = response
    for name in ("usage", *path):
        value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def default_temperature_for_model(model: str) -> float | None:
    normalized = model.lower()
    if "gemini-3" in normalized:
        return None
    return 0.1
