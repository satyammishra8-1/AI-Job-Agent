"""One live assisted-apply session: the CLI's review gate, served over HTTP.

An :class:`ApplySession` owns a VISIBLE browser (headless is never used here —
the human must be able to log in, solve captchas, and answer consent boxes in
the window) and mirrors the CLI runner's safety structure exactly:

* fields are filled with the same ``build_fill_plan`` → ``apply_plan`` path;
  a failed fill is demoted back to unfilled, never fatal;
* consent/legal/EEO fields are never auto-filled here — ``mark_done`` only
  records that the HUMAN completed one in the browser, the page is untouched;
* :meth:`submit` is the ONLY path to a real submission, refuses while any
  required field is unfilled (the CLI's approval rule), and routes through
  ``run_submit`` — the same two-lock gate the terminal flow uses;
* every outcome lands in the tracker (submitted / paused).

Playwright's sync API is bound to the thread that created it, while FastAPI
serves each request from a pool thread — so the session runs a dedicated
worker thread that owns the browser, and public methods marshal calls into it.
"""

from __future__ import annotations

import threading
import hashlib
import json
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue

from job_agent.apply.browser import open_browser
from job_agent.apply.fields import FillPlan, PlannedFill
from job_agent.apply.filler import apply_plan, build_fill_plan
from job_agent.apply.form_reader import read_form
from job_agent.apply.review import Decision, ReviewOutcome
from job_agent.apply.runner import _SUBMIT_SELECTOR, _TRACK_STATUS
from job_agent.apply.screening import apply_drafts
from job_agent.apply.submit import log_result, run_submit
from job_agent.apply.tracker import (ApplicationRecord, record_attempt,
                                    update_application, update_status)
from job_agent.dashboard.service import serialize_plan
from job_agent.store import resolve_apply_url, validate_application_url

_CALL_TIMEOUT_S = 600   # a single step may include slow page loads / an LLM draft

_UNREADABLE = (
    "Could not read an application form on this page (0 fields found). Some "
    "sources gate their form behind a captcha, login, or an extra “Apply” "
    "click — do that in the browser window that just opened, then press "
    "“Re-read form”."
)

_WINDOW_CLOSED = (
    "The browser window was closed — nothing was submitted. Cancel this "
    "session and click Apply again to reopen it."
)


def _is_page_closed(exc: Exception) -> bool:
    """True for Playwright's closed-target errors (window closed mid-session).

    Detected by class name and message so this module never imports
    Playwright — the tests drive sessions with fake pages.
    """
    if type(exc).__name__ == "TargetClosedError":
        return True
    return "has been closed" in str(exc)


class SubmitBlocked(RuntimeError):
    """Submission refused — the review gate is not satisfied."""


class SessionClosed(RuntimeError):
    """The session's browser is already torn down."""


class ApplySession:
    """A live, human-gated apply flow bound to one visible browser window."""

    def __init__(self, *, record: dict, bank, contact, resume_path: Path | None,
                 drafter, tracker_path: Path, out_dir: Path,
                 browser_factory=None, submit_selector: str = _SUBMIT_SELECTOR,
                 verification_reviewed: bool = False, preparation: dict | None = None):
        verification = record.get("verification") or {}
        if (isinstance(verification, dict)
                and verification.get("status") == "suspicious"
                and not verification_reviewed):
            raise ValueError(
                "suspicious job requires explicit human review before opening an application")
        self._record = record
        self._preparation = preparation or {}
        self._verification_reviewed = verification_reviewed
        self._bank, self._contact = bank, contact
        self._resume_path, self._drafter = resume_path, drafter
        self._tracker_path, self._out_dir = Path(tracker_path), Path(out_dir)
        self._browser_factory = browser_factory or open_browser
        self._submit_selector = submit_selector

        self._page = None
        self._plan = FillPlan()
        self._form_readable = False
        self._message = ""
        self._warnings: list[str] = []
        self._attempt_id = ""
        self._closed = False
        self._last_review_digest = ""

        self._commands: Queue = Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"apply-{record.get('id', '?')}")
        self._thread.start()

    # -- public API (each call runs inside the worker thread) -------------------

    def start(self) -> dict:
        return self._invoke(self._start)

    def rescan(self) -> dict:
        return self._invoke(self._rescan)

    def edit(self, selector: str, value: str, mark_done: bool = False) -> dict:
        return self._invoke(self._edit, selector, value, mark_done)

    def submit(self, approval_token: str) -> dict:
        return self._invoke(self._submit, approval_token)

    def cancel(self) -> dict:
        return self._invoke(self._cancel)

    # -- worker-thread plumbing --------------------------------------------------

    def _invoke(self, fn, *args):
        if self._closed:
            raise SessionClosed("this apply session is already closed")
        box: Queue = Queue()
        self._commands.put((fn, args, box))
        ok, value = box.get(timeout=_CALL_TIMEOUT_S)
        if not ok:
            raise value
        return value

    def _loop(self) -> None:
        with ExitStack() as stack:
            while True:
                fn, args, box = self._commands.get()
                try:
                    box.put((True, fn(stack, *args)))
                except Exception as exc:  # marshaled to the caller, loop survives
                    box.put((False, exc))
                if self._closed:
                    return  # ExitStack tears the browser down

    # -- steps (worker thread only) -----------------------------------------------

    def _start(self, stack: ExitStack) -> dict:
        verification = self._record.get("verification") or {}
        if (isinstance(verification, dict)
                and verification.get("status") == "suspicious"
                and not self._verification_reviewed):
            raise ValueError(
                "suspicious job requires explicit human review before opening an application")
        self._attempt_id = record_attempt(self._tracker_path, ApplicationRecord(
            company=self._record.get("company", "Unknown"),
            title=self._record.get("title", ""),
            job_id=str(self._record.get("id", "")),
            date=datetime.now(timezone.utc).isoformat(),
            source=self._record.get("source", ""),
            status="paused", reason="in dashboard review",
            application_id=uuid.uuid4().hex,
            application_url=resolve_apply_url(self._record) or "",
            verification_status=(self._record.get("verification") or {}).get("status", ""),
            eligibility_status=(self._record.get("eligibility") or {}).get("status", ""),
            match_score=self._record.get("score"),
            resume_identity=(hashlib.sha256(self._resume_path.read_bytes()).hexdigest()
                             if self._resume_path and self._resume_path.exists() else ""),
            resume_version=self._preparation.get("context_fingerprint", ""),
            answers_identity=self._preparation.get("approved_fingerprint", ""),
            answers_version=self._preparation.get("approved_fingerprint", ""),
            submitted_answers={a.get("key", ""): a.get("answer", "")
                               for a in self._preparation.get("answers", [])},
            recruiter_message=self._preparation.get("recruiter_message", ""),
            cover_note=self._preparation.get("cover_note", ""),
            approval_identity=self._preparation.get("approved_fingerprint", ""),
            approval_state="approved" if self._preparation.get("approved_fingerprint") else "",
            timeline=[{"event": "Application review opened",
                       "at": datetime.now(timezone.utc).isoformat()}]))
        # VISIBLE by definition: the human acts in this window (captcha, consent).
        self._page = stack.enter_context(self._browser_factory(headless=False))
        start_url = validate_application_url(resolve_apply_url(self._record) or "")
        self._page.goto(start_url, wait_until="load")
        self._validate_live_destination(start_url)
        self._read_and_fill()
        return self._state()

    def _rescan(self, stack: ExitStack) -> dict:
        """Re-read the CURRENT page — used after the human clears a captcha or
        clicks through to the real form in the visible window."""
        self._read_and_fill()
        return self._state()

    def _read_and_fill(self) -> None:
        self._warnings = []
        try:
            self._scan_and_fill()
        except Exception as exc:
            # the human closed the window mid-session: report it as a state,
            # not a stack trace — they cancel and start a fresh apply
            if not _is_page_closed(exc):
                raise
            self._plan, self._form_readable = FillPlan(), False
            self._message = _WINDOW_CLOSED

    def _scan_and_fill(self) -> None:
        fields = read_form(self._page)
        if not fields:
            self._plan, self._form_readable = FillPlan(), False
            src = self._record.get("source", "")
            self._message = _UNREADABLE + (
                f" (source: {src} — SmartRecruiters is known to captcha-gate "
                f"its apply flow)" if src == "sr-search" else "")
            return
        plan = build_fill_plan(fields, self._bank, self._contact, self._resume_path)
        if self._drafter is not None:
            plan = apply_drafts(plan, self._drafter)
        for pf in apply_plan(self._page, plan):
            plan = plan.demote(pf.field.selector,
                               "could not be filled on the live page — "
                               "complete it in the browser")
            self._warnings.append(f"could not fill: {pf.field.describe()}")
        self._plan, self._form_readable, self._message = plan, True, ""

    def _edit(self, stack: ExitStack, selector: str, value: str,
              mark_done: bool) -> dict:
        source = ("manual (completed in the browser)" if mark_done
                  else "manual (edit)")
        self._plan = self._plan.with_value(selector, value, source)
        if not mark_done:
            fill = next(p for p in self._plan.planned
                        if p.field.selector == selector)
            for failed in apply_plan(self._page, FillPlan(planned=(fill,))):
                self._warnings.append(
                    f"edit did not land on the page: {failed.field.describe()} — "
                    f"set it in the browser window")
        return self._state()

    def _browser_state(self):
        """Capture the live destination, form controls, values, and validity."""
        if self._page is None:
            raise SubmitBlocked("application page is not available")
        try:
            return self._page.evaluate("""() => ({
          url: location.href,
          page_text: (document.body && document.body.innerText || '').slice(0, 100000),
          forms_html: Array.from(document.forms).map(f => f.outerHTML),
          controls: Array.from(document.querySelectorAll(
            'input,select,textarea,[role="combobox"],[role="checkbox"],[role="radio"],[contenteditable="true"]'
          )).map(e => ({
            name: e.name || '', id: e.id || '', type: e.type || '',
            value: e.value || '', checked: !!e.checked,
            selected: e.tagName === 'SELECT' ? e.value : '',
            label: e.getAttribute('aria-label') || '',
            placeholder: e.getAttribute('placeholder') || '',
            required: !!e.required,
            text: e.isContentEditable ? e.innerText : ''
          })),
          invalid: Array.from(document.querySelectorAll('input,select,textarea'))
            .filter(e => typeof e.checkValidity === 'function' && !e.checkValidity())
            .map(e => e.id || e.name || e.type || 'required control'),
          forms: Array.from(document.forms).map(f => ({
            action: f.action,
            valid: f.checkValidity()
          }))
            })""")
        except Exception as exc:
            if _is_page_closed(exc):
                raise SubmitBlocked("the browser window was closed; nothing was submitted") from exc
            raise

    def _validate_live_destination(self, expected_url: str | None = None) -> None:
        from urllib.parse import urlsplit

        try:
            current = validate_application_url(getattr(self._page, "url", expected_url) or "")
        except ValueError as exc:
            raise SubmitBlocked("application page has an unsafe or malformed destination") from exc
        if expected_url:
            expected = urlsplit(expected_url)
            actual = urlsplit(current)
            if (actual.scheme, actual.hostname, actual.port) != (expected.scheme, expected.hostname, expected.port):
                raise SubmitBlocked("application page redirected to a different destination; review it manually")
        state = self._browser_state()
        for form in state.get("forms", []) if isinstance(state, dict) else []:
            action = form.get("action")
            if action:
                try:
                    safe_action = validate_application_url(action)
                except ValueError as exc:
                    raise SubmitBlocked("application form has an unsafe or malformed destination") from exc
                parsed_action, parsed_page = urlsplit(safe_action), urlsplit(current)
                if (parsed_action.scheme, parsed_action.hostname, parsed_action.port) != (
                        parsed_page.scheme, parsed_page.hostname, parsed_page.port):
                    raise SubmitBlocked("application form submits to a different destination; review it manually")

    def _review_digest(self) -> str:
        original = validate_application_url(resolve_apply_url(self._record) or "")
        self._validate_live_destination(original)
        state = {
            "job_id": str(self._record.get("id", "")),
            "company": self._record.get("company", ""),
            "title": self._record.get("title", ""),
            "source": self._record.get("source", ""),
            "apply_url": self._record.get("apply_url") or self._record.get("url", ""),
            "plan": serialize_plan(self._plan),
            "browser": self._browser_state(),
        }
        payload = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _submit(self, stack: ExitStack, approval_token: str) -> dict:
        if not self._form_readable:
            raise SubmitBlocked("application form is not readable; re-read and review it before submitting")
        if not approval_token or approval_token != self._last_review_digest:
            self._last_review_digest = ""
            raise SubmitBlocked("review state changed or approval expired; review the form again")
        current_digest = self._review_digest()
        if current_digest != self._last_review_digest:
            self._last_review_digest = ""
            raise SubmitBlocked("application changed after review; re-read and approve it again")
        browser_state = self._browser_state()
        if isinstance(browser_state, dict) and browser_state.get("invalid"):
            self._last_review_digest = ""
            raise SubmitBlocked("the live form has missing or invalid required fields")
        if isinstance(browser_state, dict) and any(
                not form.get("valid", True) for form in browser_state.get("forms", [])):
            self._last_review_digest = ""
            raise SubmitBlocked("the live form has missing or invalid required fields")
        missing = self._plan.missing_required()
        if missing:
            self._last_review_digest = ""
            names = ", ".join(u.field.describe() for u in missing[:4])
            raise SubmitBlocked(
                f"{len(missing)} required field(s) still unfilled ({names}) — "
                f"edit them or mark them done first")
        outcome = ReviewOutcome(Decision.APPROVE, self._plan)
        self._last_review_digest = ""  # one approval authorizes exactly one attempt
        try:
            result = run_submit(
                self._page, self._plan, outcome, submit_flag=True,
                submit_selector=self._submit_selector, screenshot_dir=self._out_dir,
                job_label=f"{self._record.get('company', '?')} — "
                          f"{self._record.get('title', '?')}")
        except Exception as exc:
            if not _is_page_closed(exc):
                raise
            raise SubmitBlocked(
                "the browser window was closed — nothing was submitted; "
                "cancel and start apply again") from exc
        log_result(result, self._out_dir / "apply_log.jsonl")
        if result.status == "submitted":
            controls = browser_state.get("controls", []) if isinstance(browser_state, dict) else []
            excluded = ("password", "otp", "one-time", "token", "captcha", "ssn", "social security")
            exact_answers = {}
            for control in controls:
                identity = " ".join(str(control.get(k, "")) for k in
                                    ("name", "id", "label", "placeholder")).strip()
                if (control.get("type", "").lower() == "password"
                        or any(term in identity.lower() for term in excluded)):
                    continue
                value = control.get("value") or control.get("text") or ""
                if value:
                    exact_answers[identity or f"field-{len(exact_answers) + 1}"] = str(value)
            update_application(self._tracker_path, self._attempt_id,
                submitted_answers=exact_answers,
                submission_approval=hashlib.sha256(approval_token.encode()).hexdigest())
        update_status(self._tracker_path, self._attempt_id,
                      _TRACK_STATUS.get(result.status, "paused"), result.reason)
        if result.status == "submitted":
            self._closed = True
        return {"result": {"status": result.status, "reason": result.reason,
                           "screenshot": result.screenshot}, **self._state()}

    def _cancel(self, stack: ExitStack) -> dict:
        if self._attempt_id:
            update_status(self._tracker_path, self._attempt_id, "paused",
                          "review cancelled in the dashboard — nothing submitted")
        self._closed = True
        return {"result": {"status": "cancelled"}}

    def _state(self) -> dict:
        try:
            digest = self._review_digest() if self._page is not None else ""
        except SubmitBlocked:
            digest = ""
        self._last_review_digest = digest
        return {
            "job": {"id": self._record.get("id"),
                    "title": self._record.get("title"),
                    "company": self._record.get("company")},
            "form_readable": self._form_readable,
            "message": self._message,
            "approval_token": digest,
            "warnings": list(self._warnings),
            **serialize_plan(self._plan),
        }
