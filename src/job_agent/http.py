"""A small shared HTTP helper for the ATS sources.

All boards are public GET endpoints. This centralizes the timeout, a couple of
retries on transient failures, a descriptive User-Agent, and error mapping so
each source doesn't reinvent it. Failures raise :class:`SourceError`, which the
sources catch and turn into "this board contributed no jobs" rather than
crashing the whole run.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any

import httpx

USER_AGENT = "job-agent/0.1 (+https://github.com/; portfolio project)"
DEFAULT_TIMEOUT = 20.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.5
# Transient statuses worth a retry; 4xx (except 429) are not retried.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_audit_sink: ContextVar[Any | None] = ContextVar("job_agent_audit_sink", default=None)
_audit_source: ContextVar[str] = ContextVar("job_agent_audit_source", default="")
_audit_phase: ContextVar[str] = ContextVar("job_agent_audit_phase", default="")


@contextmanager
def audit_request_scope(sink, source: str, phase: str):
    """Collect per-source request diagnostics only for an explicit audit run."""
    a = _audit_sink.set(sink)
    b = _audit_source.set(source)
    c = _audit_phase.set(phase)
    try:
        yield
    finally:
        _audit_phase.reset(c)
        _audit_source.reset(b)
        _audit_sink.reset(a)


class SourceError(RuntimeError):
    """A board could not be fetched (network, HTTP, or malformed JSON)."""


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    _sleep=time.sleep,
) -> Any:
    """GET ``url`` and return parsed JSON, retrying transient failures.

    Raises :class:`SourceError` on any unrecoverable problem.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        status = None
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
            status = resp.status_code
        except httpx.HTTPError as exc:  # connection/timeout/etc.
            last_error = exc
        else:
            if resp.status_code == 200:
                sink = _audit_sink.get()
                if sink is not None:
                    sink.record_request(_audit_source.get(), _audit_phase.get(), status,
                                        time.monotonic() - started, None)
                try:
                    return resp.json()
                except ValueError as exc:
                    raise SourceError(f"{url} returned non-JSON body") from exc
            if resp.status_code not in RETRYABLE_STATUS:
                sink = _audit_sink.get()
                if sink is not None:
                    sink.record_request(_audit_source.get(), _audit_phase.get(), status,
                                        time.monotonic() - started,
                                        f"HTTP {resp.status_code}")
                raise SourceError(f"{url} -> HTTP {resp.status_code}")
            last_error = SourceError(f"{url} -> HTTP {resp.status_code}")

        sink = _audit_sink.get()
        if sink is not None:
            sink.record_request(
                _audit_source.get(), _audit_phase.get(), status,
                time.monotonic() - started,
                None if status == 200 else str(last_error or "request failed"),
            )

        if attempt < MAX_ATTEMPTS:
            _sleep(BACKOFF_SECONDS * attempt)

    raise SourceError(f"{url} failed after {MAX_ATTEMPTS} attempts: {last_error}")
