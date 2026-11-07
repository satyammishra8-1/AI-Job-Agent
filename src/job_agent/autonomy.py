"""Persistent policy and scheduling primitives for autonomous job discovery.

Submission remains fail-closed: this module only authorizes candidates that
meet an explicit standing policy. Browser/session blockers are recorded as
paused work and are never treated as a successful application.
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from job_agent.json_store import read_json, write_json_atomic

_POLICY_LOCK = threading.RLock()


def submitted_today(records, *, day: str) -> int:
    """Count submissions by receipt date even after the tracker status advances."""
    counted = {"submitted", "applied", "under_review", "assessment", "interviewing", "offer"}
    def local_day(record) -> str:
        value = str(record.applied_at or "")
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return stamp.astimezone().date().isoformat()
        except ValueError:
            return value[:10]
    return sum(1 for record in records if record.status in counted and local_day(record) == day)


class AutonomousPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    authorized_at: str = ""
    max_applications_per_day: int = Field(default=5, ge=1, le=100)
    min_match_score: int = Field(default=70, ge=0, le=100)
    allowed_locations: list[str] = Field(default_factory=list)
    allowed_work_modes: list[str] = Field(default_factory=lambda: ["remote", "hybrid", "onsite"])
    allowed_durations: list[str] = Field(default_factory=list)
    minimum_stipend: float | None = Field(default=None, ge=0)
    graduation_years: list[int] = Field(default_factory=list)
    academic_years: list[str] = Field(default_factory=list)
    branches: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    interval_minutes: int = Field(default=360, ge=30, le=10080)


def load_policy(path: str | Path) -> AutonomousPolicy:
    with _POLICY_LOCK:
        raw = read_json(path, {})
        try:
            return AutonomousPolicy.model_validate(raw if isinstance(raw, dict) else {})
        except Exception:
            # Corrupt settings disable automation rather than silently restoring opt-in.
            return AutonomousPolicy()


def save_policy(path: str | Path, policy: AutonomousPolicy) -> None:
    with _POLICY_LOCK:
        write_json_atomic(path, policy.model_dump(mode="json"))


def authorization_update(policy: AutonomousPolicy, *, enabled: bool,
                         now: datetime | None = None) -> AutonomousPolicy:
    """Enabling records a deliberate opt-in; disabling is always immediate."""
    stamp = (now or datetime.now(timezone.utc)).isoformat() if enabled else ""
    return policy.model_copy(update={"enabled": enabled, "authorized_at": stamp})


def idempotency_key(job: dict) -> str:
    identity = "|".join(str(job.get(k, "")).strip().lower()
                        for k in ("source", "id", "company", "title"))
    return hashlib.sha256(identity.encode()).hexdigest()


def evaluate_policy(job: dict, profile: dict, policy: AutonomousPolicy) -> tuple[str, list[str]]:
    """Return ELIGIBLE/REVIEW/NOT_ELIGIBLE for auto-processing.

    Missing facts are REVIEW. Hard mismatches and failed configured constraints
    are NOT_ELIGIBLE. The caller must also require verified provenance and the
    existing deterministic eligibility engine's ELIGIBLE result.
    """
    reasons: list[str] = []
    if not policy.enabled or not policy.authorized_at:
        return "REVIEW", ["Automatic applications are not explicitly authorized."]
    if (job.get("verification") or {}).get("status") != "verified":
        return "REVIEW", ["Employer and posting are not verified."]
    if (job.get("eligibility") or {}).get("status") != "eligible":
        return "REVIEW", ["Profile eligibility is not established."]
    score = job.get("score")
    if score is None:
        return "REVIEW", ["Match score is unavailable."]
    if score < policy.min_match_score:
        return "NOT_ELIGIBLE", [f"Match score {score} is below the configured minimum."]
    title = str(job.get("title", ""))
    roles = policy.target_roles or profile.get("target_roles", [])
    if roles and not any(role.casefold() in title.casefold() for role in roles):
        return "NOT_ELIGIBLE", ["Role does not match configured target roles."]
    location = str(job.get("location", ""))
    allowed_locations = policy.allowed_locations or profile.get("preferred_locations", [])
    if allowed_locations and not any(loc.casefold() in location.casefold() for loc in allowed_locations):
        return "NOT_ELIGIBLE", ["Location is outside the configured allow-list."]
    remote = job.get("remote")
    mode = "remote" if remote is True else "onsite" if remote is False else None
    if mode is None and "hybrid" in location.casefold():
        mode = "hybrid"
    if mode is None:
        return "REVIEW", ["Work mode is not stated clearly."]
    if mode not in policy.allowed_work_modes:
        return "NOT_ELIGIBLE", [f"Work mode {mode} is not allowed."]
    profile_mode = str(profile.get("remote_preference") or "any").casefold()
    if profile_mode not in {"", "any"} and mode != profile_mode:
        return "NOT_ELIGIBLE", [f"Work mode {mode} conflicts with the profile preference ({profile_mode})."]
    facts = profile
    if policy.graduation_years:
        year = facts.get("expected_graduation")
        if year is None:
            return "REVIEW", ["Expected graduation year is missing from the profile."]
        if year not in policy.graduation_years:
            return "NOT_ELIGIBLE", ["Expected graduation year is outside the configured range."]
    if policy.academic_years:
        year = facts.get("current_year")
        if not year:
            return "REVIEW", ["Current academic year is missing from the profile."]
        if str(year).casefold() not in {x.casefold() for x in policy.academic_years}:
            return "NOT_ELIGIBLE", ["Academic year is outside the configured allow-list."]
    if policy.branches:
        education = " ".join(str(x) for x in facts.get("education", [])) + " " + " ".join(
            str(x.get("field", "")) for x in facts.get("education_details", []))
        if not education.strip():
            return "REVIEW", ["Degree branch is missing from the profile."]
        if not any(branch.casefold() in education.casefold() for branch in policy.branches):
            return "NOT_ELIGIBLE", ["Degree branch is outside the configured allow-list."]
    if policy.allowed_durations:
        duration = str(job.get("internship_duration") or job.get("duration") or "")
        if not duration:
            import re
            match = re.search(r"\b(\d{1,2})\s*(months?|weeks?)\b",
                              str(job.get("description", "")), re.IGNORECASE)
            duration = match.group(0) if match else ""
        if not duration:
            return "REVIEW", ["Internship duration is not stated in the posting."]
        if not any(x.casefold() in duration.casefold() for x in policy.allowed_durations):
            return "NOT_ELIGIBLE", ["Internship duration is outside the configured allow-list."]
    if policy.minimum_stipend is not None:
        stipend = job.get("stipend")
        if stipend is None:
            return "REVIEW", ["Stipend is not stated in the posting."]
        if float(stipend) < policy.minimum_stipend:
            return "NOT_ELIGIBLE", ["Stipend is below the configured minimum."]
    return "ELIGIBLE", ["All configured autonomous policy checks passed."]


class AutonomousQueue:
    """Atomic JSON queue with durable idempotency and attempt history."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self) -> dict:
        raw = read_json(self.path, {})
        return raw if isinstance(raw, dict) else {}

    def enqueue(self, job: dict, status: str, reasons: list[str], *, now=None) -> bool:
        key = idempotency_key(job)
        with self.lock:
            data = self.read()
            jobs = data.setdefault("jobs", {})
            if key in jobs:
                return False
            jobs[key] = {"key": key, "job_id": str(job.get("id", "")),
                         "company": job.get("company", ""), "title": job.get("title", ""),
                         "status": status, "decision": status, "reasons": reasons,
                         "created_at": (now or datetime.now(timezone.utc)).isoformat(),
                         "attempts": []}
            write_json_atomic(self.path, data)
            return True

    def record(self, key: str, status: str, reason: str, *, now=None) -> None:
        with self.lock:
            data = self.read()
            item = data.setdefault("jobs", {}).get(key)
            if item is None:
                raise KeyError(key)
            stamp = (now or datetime.now(timezone.utc)).isoformat()
            item["status"] = status
            item.setdefault("attempts", []).append({"status": status, "reason": reason, "at": stamp})
            write_json_atomic(self.path, data)

    def daily_submissions(self, *, now=None) -> int:
        today = (now or datetime.now(timezone.utc)).date()
        return sum(1 for item in self.read().get("jobs", {}).values()
                   if item.get("status") == "submitted" and item.get("submitted_at", "")[:10] == today.isoformat())


class DiscoveryScheduler:
    """Single-worker scheduler; persisted timestamps support safe restart."""
    def __init__(self, state_path: str | Path, policy_path: str | Path,
                 callback, *, clock=time.time):
        self.state_path, self.policy_path = Path(state_path), Path(policy_path)
        self.callback, self.clock = callback, clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            # Resume a worker paused during a long in-flight scan instead of
            # starting a second worker or leaving enabled mode unscheduled.
            self._stop.clear()
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True, name="job-discovery-scheduler")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            policy = load_policy(self.policy_path)
            state = read_json(self.state_path, {})
            now = self.clock()
            due = not state.get("next_run") or float(state["next_run"]) <= now
            if policy.enabled and due:
                # Persist before work to prevent duplicate workers/restart storms.
                state["last_run"] = now
                state["next_run"] = now + policy.interval_minutes * 60
                write_json_atomic(self.state_path, state)
                try:
                    self.callback()
                except Exception as exc:
                    state["last_error"] = f"{type(exc).__name__}: {exc}"
                    write_json_atomic(self.state_path, state)
            self._stop.wait(1.0 if not policy.enabled else min(60.0, policy.interval_minutes * 60))
