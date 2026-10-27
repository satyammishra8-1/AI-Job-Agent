"""Application tracking: a gitignored JSON log of every apply attempt.

``data/applications.json`` (already covered by the ``/data/*`` gitignore — it
holds personal job history) records one entry per ``run_apply`` attempt:
company, role, job id, ISO date, source ATS, and a status that resolves to
``submitted`` / ``paused`` / ``failed`` when the run completes.

Everything is immutable in the codebase style: records are frozen models and
updates rewrite the file with a new list rather than mutating in place. A
missing or corrupt log reads as empty — tracking must never break an apply run.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from job_agent.json_store import read_json, write_json_atomic

#: Run statuses (written by run_apply): submitted — really sent; paused —
#: awaiting the human (dry-run, skipped, mid-run); failed — the run crashed.
#: User statuses (set from the dashboard): saved → ready_to_apply → applied →
#: under_review / assessment / interviewing → offer / rejected / withdrawn.
Status = Literal["submitted", "paused", "failed", "saved", "ready_to_apply",
                 "applied", "under_review", "assessment", "interviewing",
                 "offer", "rejected", "withdrawn"]

_TRANSITIONS = {
    "saved": {"ready_to_apply", "applied", "rejected", "withdrawn"},
    "ready_to_apply": {"applied", "submitted", "rejected", "withdrawn"},
    "submitted": {"applied", "under_review", "assessment", "interviewing", "offer", "rejected", "withdrawn"},
    "applied": {"under_review", "assessment", "interviewing", "offer", "rejected", "withdrawn"},
    "under_review": {"assessment", "interviewing", "offer", "rejected", "withdrawn"},
    "assessment": {"under_review", "interviewing", "offer", "rejected", "withdrawn"},
    "interviewing": {"assessment", "interviewing", "offer", "rejected", "withdrawn"},
    "offer": {"rejected", "withdrawn"},
    "rejected": set(), "withdrawn": set(),
    "paused": {"ready_to_apply", "applied", "submitted", "under_review", "assessment",
               "interviewing", "offer", "rejected", "failed", "withdrawn"},
    "failed": {"ready_to_apply", "rejected", "withdrawn"},
}


class ApplicationRecord(BaseModel):
    """One tracked job/application as written to the log."""

    model_config = ConfigDict(frozen=True)

    company: str
    title: str = ""
    job_id: str = ""
    date: str                    # ISO-8601 UTC timestamp of the attempt
    source: str = ""             # ATS the job came from (greenhouse/ashby/...)
    status: Status
    reason: str = ""             # human-readable outcome context
    attempt_id: str = ""         # unique per attempt; assigned by record_attempt
    # user-managed state (edited from the dashboard, never by run_apply)
    notes: str = ""              # recruiter name, why applied, ...
    follow_up: str = ""          # ISO date to follow up by ("" = none)
    application_id: str = ""
    application_url: str = ""
    verification_status: str = ""
    eligibility_status: str = ""
    match_score: float | None = None
    profile_id: str = ""
    profile_version: int = 0
    resume_identity: str = ""
    resume_version: str = ""
    answers_identity: str = ""
    answers_version: str = ""
    submitted_answers: dict[str, str] = {}
    recruiter_message: str = ""
    cover_note: str = ""
    approval_identity: str = ""
    submission_approval: str = ""
    approval_state: str = ""
    applied_at: str = ""
    follow_up_note: str = ""
    follow_up_completed: bool = False
    interview_date: str = ""
    interview_type: str = ""
    interview_notes: str = ""
    interview_outcome: str = ""
    assessment_date: str = ""
    assessment_type: str = ""
    assessment_notes: str = ""
    assessment_outcome: str = ""
    offer_date: str = ""
    offer_notes: str = ""
    offer_outcome: str = ""
    rejection_date: str = ""
    rejection_reason: str = ""
    rejection_notes: str = ""
    timeline: list[dict] = []
    updated_at: str = ""


def transition_allowed(old: str, new: str) -> bool:
    return old == new or new in _TRANSITIONS.get(old, set())


def load_applications(path: str | Path) -> list[ApplicationRecord]:
    """Read the log; a missing file is empty, corrupt data is an error."""
    path = Path(path)
    try:
        raw = read_json(path, [])
        if not isinstance(raw, list):
            raise ValueError(f"Invalid application log in {path}: expected a JSON list")
        return [ApplicationRecord.model_validate(r) for r in raw]
    except ValidationError as exc:
        raise ValueError(f"Invalid application record in {path}: {exc}") from exc


def _write(path: Path, records: list[ApplicationRecord]) -> None:
    write_json_atomic(path, [r.model_dump() for r in records])


def _find_job(records: list[ApplicationRecord], job_id: str,
              company: str, title: str) -> int | None:
    """Index of the record tracking this JOB: exact id first, then the
    (company, title) match-key — sr-search re-issues ids across runs, so the
    same posting must be recognized under a new id."""
    if job_id:
        idx = next((i for i in range(len(records) - 1, -1, -1)
                    if records[i].job_id == str(job_id)), None)
        if idx is not None:
            return idx
    if company and title:
        key = (_match_key(company), _match_key(title))
        return next((i for i in range(len(records) - 1, -1, -1)
                     if (_match_key(records[i].company),
                         _match_key(records[i].title)) == key), None)
    return None


def record_attempt(path: str | Path, record: ApplicationRecord) -> str:
    """Track one attempt — ONE record per job, never a duplicate row.

    A new attempt on an already-tracked job (matched by id, else by
    company+title) REPLACES that record: the new attempt's status/date/id win,
    the user's notes and follow-up survive. Returns the unique attempt id.
    """
    path = Path(path)
    records = load_applications(path)
    attempt_id = record.attempt_id or uuid.uuid4().hex
    stamped = record.model_copy(update={
        "attempt_id": attempt_id,
        "application_id": record.application_id or uuid.uuid4().hex,
        "updated_at": record.updated_at or record.date,
        "timeline": record.timeline or [{"status": record.status, "at": record.date}],
    })
    idx = _find_job(records, record.job_id, record.company, record.title)
    if idx is None:
        _write(path, records + [stamped])
    else:
        merged = stamped.model_copy(update={"notes": records[idx].notes,
                                            "follow_up": records[idx].follow_up})
        _write(path, records[:idx] + [merged] + records[idx + 1:])
    return attempt_id


def update_status(path: str | Path, attempt_id: str, status: Status,
                  reason: str = "") -> None:
    """Set the final status of one attempt (new records list — no mutation)."""
    path = Path(path)
    records = load_applications(path)
    for i, r in enumerate(records):
        if r.attempt_id != attempt_id:
            continue
        if not transition_allowed(r.status, status):
            raise ValueError(f"Invalid application status transition: {r.status} -> {status}")
        now = datetime.now(timezone.utc).isoformat()
        event = {"status": status, "at": now}
        records[i] = r.model_copy(update={"status": status, "reason": reason or r.reason,
                                         "updated_at": now,
                                         "applied_at": now if status in {"applied", "submitted"} else r.applied_at,
                                         "timeline": [*r.timeline, event]})
        break
    _write(path, records)


def record_event(path: str | Path, attempt_id: str, event: str, **fields) -> ApplicationRecord:
    """Append a real user-recorded event and update its associated details."""
    records = load_applications(path)
    for i, record in enumerate(records):
        if record.attempt_id == attempt_id:
            now = datetime.now(timezone.utc).isoformat()
            updated = record.model_copy(update={**fields, "updated_at": now,
                "applied_at": now if fields.get("status") == "applied" else record.applied_at,
                "timeline": [*record.timeline, {"event": event, "at": now}]})
            records[i] = updated
            _write(Path(path), records)
            return updated
    raise KeyError(attempt_id)


def update_application(path: str | Path, attempt_id: str, **fields) -> ApplicationRecord:
    """Persist exact application integrity details on an existing record."""
    records = load_applications(path)
    for i, record in enumerate(records):
        if record.attempt_id == attempt_id:
            updated = record.model_copy(update={**fields,
                "updated_at": datetime.now(timezone.utc).isoformat()})
            records[i] = updated
            _write(Path(path), records)
            return updated
    raise KeyError(attempt_id)


#: A job with one of these statuses has an application in flight — search views
#: hide it (re-applying would be a duplicate). Saved, ready, paused, rejected,
#: failed, and withdrawn records remain visible for review or retry.
APPLIED_STATUSES = frozenset({
    "applied", "submitted", "under_review", "assessment", "interviewing", "offer",
})


def _match_key(text: str) -> str:
    """Loose identity for cross-run matching: lowercase, alnum words only."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def applied_markers(path: str | Path) -> dict:
    """The identity sets of every in-flight application, for exclusion checks.

    ``ids`` match exactly; ``pairs`` are (company, title) match-keys so the same
    role is recognized across search runs even when the source re-issues ids
    (sr-search does).
    """
    ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for r in load_applications(path):
        if r.status in APPLIED_STATUSES:
            if r.job_id:
                ids.add(str(r.job_id))
            if r.company and r.title:
                pairs.add((_match_key(r.company), _match_key(r.title)))
    return {"ids": ids, "pairs": pairs}


def is_already_applied(markers: dict, *, job_id: str, company: str, title: str) -> bool:
    """True when this job already has an in-flight application (id or re-match)."""
    if str(job_id) in markers["ids"]:
        return True
    return bool(company and title
                and (_match_key(company), _match_key(title)) in markers["pairs"])


def upsert_job_state(path: str | Path, *, job_id: str = "", attempt_id: str = "",
                     company: str = "", title: str = "", source: str = "",
                     status: Status | None = None, notes: str | None = None,
                     follow_up: str | None = None,
                     follow_up_note: str | None = None,
                     follow_up_completed: bool | None = None,
                     now: datetime | None = None) -> ApplicationRecord:
    """Set user-managed state for one job, creating its record if none exists.

    Targets the MOST RECENT record with this ``job_id`` — or, for records that
    carry no job id (old/manual entries), the record with ``attempt_id``. A job
    never tracked before gets a fresh record (default status "saved"). Only the
    fields actually passed are changed — None means "leave as is". Immutable
    style: the file is rewritten with a new records list.
    """
    if not job_id and not attempt_id:
        raise ValueError("upsert_job_state needs a job_id or an attempt_id")
    path = Path(path)
    records = load_applications(path)
    if attempt_id and not job_id:
        idx = next((i for i, r in enumerate(records)
                    if r.attempt_id == attempt_id), None)
        if idx is None:
            raise KeyError(f"no record with attempt_id {attempt_id!r}")
    else:
        # id first, then company+title — the same job under a re-issued id
        # must update its existing row, never grow a duplicate
        idx = _find_job(records, job_id, company, title)
    if idx is None:
        record = ApplicationRecord(
            company=company or "Unknown", title=title, job_id=job_id,
            date=(now or datetime.now(timezone.utc)).isoformat(), source=source,
            status=status or "saved", notes=notes or "", follow_up=follow_up or "",
            follow_up_note=follow_up_note or "",
            follow_up_completed=follow_up_completed or False,
            application_id=uuid.uuid4().hex,
            updated_at=(now or datetime.now(timezone.utc)).isoformat(),
            timeline=[{"status": status or "saved",
                       "at": (now or datetime.now(timezone.utc)).isoformat()}],
            attempt_id=uuid.uuid4().hex)
        _write(path, records + [record])
        return record
    updates = {k: v for k, v in
               (("status", status), ("notes", notes), ("follow_up", follow_up),
                ("follow_up_note", follow_up_note),
                ("follow_up_completed", follow_up_completed))
               if v is not None}
    if updates:
        updates["updated_at"] = (now or datetime.now(timezone.utc)).isoformat()
        if follow_up is not None and follow_up != records[idx].follow_up:
            updates["follow_up_completed"] = False
    record = records[idx].model_copy(update=updates)
    if status is not None and status != records[idx].status:
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        record = record.model_copy(update={
            "timeline": [*record.timeline, {"status": status, "at": stamp}],
            "applied_at": stamp if status == "applied" else record.applied_at,
        })
    _write(path, records[:idx] + [record] + records[idx + 1:])
    return record
