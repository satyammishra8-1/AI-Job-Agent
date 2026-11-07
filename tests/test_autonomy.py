from datetime import datetime, timezone
from threading import Event
import time

from fastapi.testclient import TestClient

from job_agent.autonomy import (AutonomousPolicy, AutonomousQueue,
    DiscoveryScheduler, authorization_update, evaluate_policy, idempotency_key,
    load_policy, save_policy, submitted_today)
from job_agent.dashboard.app import create_app
from job_agent.apply.fields import FieldType, FillPlan, FormField, Unfilled
from job_agent.dashboard.apply_session import automatic_block_reason


def eligible_job(**updates):
    job = {"id": "42", "source": "greenhouse", "company": "Example",
           "title": "Software Intern", "location": "Remote", "remote": True,
           "duration": "3 months",
           "score": 85, "verification": {"status": "verified"},
           "eligibility": {"status": "eligible"}}
    return {**job, **updates}


def test_policy_defaults_off_and_requires_explicit_authorization():
    policy = AutonomousPolicy()
    assert not policy.enabled and not policy.authorized_at
    enabled = authorization_update(policy, enabled=True,
        now=datetime(2026, 11, 7, tzinfo=timezone.utc))
    assert enabled.enabled and enabled.authorized_at
    assert not authorization_update(enabled, enabled=False).authorized_at


def test_policy_accepts_only_complete_profile_evidence():
    policy = authorization_update(AutonomousPolicy(), enabled=True)
    facts = {"expected_graduation": 2027, "current_year": "3rd year",
             "education_details": [{"field": "Computer Science"}],
             "internship_duration": "3 months"}
    policy = policy.model_copy(update={"graduation_years": [2027],
        "academic_years": ["3rd year"], "branches": ["Computer Science"],
        "allowed_durations": ["3 months"]})
    assert evaluate_policy(eligible_job(), facts, policy)[0] == "ELIGIBLE"
    assert evaluate_policy(eligible_job(), {}, policy)[0] == "REVIEW"
    assert evaluate_policy(eligible_job(remote=False), facts,
        policy.model_copy(update={"allowed_work_modes": ["remote"]}))[0] == "NOT_ELIGIBLE"
    assert evaluate_policy(eligible_job(remote=False), {**facts, "remote_preference": "remote"},
        policy.model_copy(update={"allowed_work_modes": ["remote", "onsite"]}))[0] == "NOT_ELIGIBLE"
    assert evaluate_policy(eligible_job(score=30), facts, policy)[0] == "NOT_ELIGIBLE"


def test_daily_limit_counts_submitted_applications_after_status_changes():
    class Record:
        def __init__(self, status, applied_at):
            self.status, self.applied_at = status, applied_at
    records = [Record("under_review", "2026-11-07T10:00:00+00:00"),
               Record("submitted", "2026-11-06T10:00:00+00:00")]
    assert submitted_today(records, day="2026-11-07") == 1


def test_live_form_blocks_captcha_legal_and_unknown_required_answers():
    empty = FillPlan()
    assert "anti-bot" in automatic_block_reason(True, empty,
        {"page_text": "Please complete CAPTCHA"})
    legal = FillPlan(unfilled=(Unfilled(FormField("consent", FieldType.CHECKBOX,
        label="I agree to the privacy consent", required=True), "legal"),))
    assert "Legal" in automatic_block_reason(True, legal, {"page_text": "Apply"})
    required = FillPlan(unfilled=(Unfilled(FormField("q", FieldType.TEXT,
        label="Unknown question", required=True), "no evidence"),))
    assert "required field" in automatic_block_reason(True, required, {"page_text": "Apply"})
    assert automatic_block_reason(True, empty, {"page_text": "Apply"}) is None


def test_queue_idempotency_and_persisted_attempts(tmp_path):
    queue = AutonomousQueue(tmp_path / "queue.json")
    job = eligible_job()
    assert queue.enqueue(job, "ELIGIBLE", ["ok"])
    assert not queue.enqueue(job, "ELIGIBLE", ["ok"])
    key = idempotency_key(job)
    queue.record(key, "PAUSED", "Login required")
    restored = AutonomousQueue(tmp_path / "queue.json")
    assert restored.read()["jobs"][key]["attempts"][0]["reason"] == "Login required"


def test_scheduler_starts_once_and_recovers_persisted_schedule(tmp_path):
    policy_path = tmp_path / "policy.json"
    save_policy(policy_path, authorization_update(AutonomousPolicy(interval_minutes=30), enabled=True))
    state_path = tmp_path / "schedule.json"
    calls = []
    started = Event()

    def callback():
        calls.append(1)
        started.set()

    scheduler = DiscoveryScheduler(state_path, policy_path, callback)
    scheduler.start(); scheduler.start()
    assert started.wait(2)
    time.sleep(.05)
    scheduler.stop()
    assert len(calls) == 1
    assert load_policy(policy_path).enabled


def test_dashboard_automatic_applications_need_one_time_opt_in(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path, searcher=lambda _days: {"ok": True}))
    assert client.get("/api/autonomous").json()["policy"]["enabled"] is False
    payload = {"enabled": True, "authorize": False}
    assert client.put("/api/autonomous", json=payload).status_code == 409
    payload["authorize"] = True
    response = client.put("/api/autonomous", json=payload)
    assert response.status_code == 200
    assert response.json()["policy"]["enabled"] is True
    payload = {"enabled": False, "authorize": False}
    assert client.put("/api/autonomous", json=payload).json()["policy"]["enabled"] is False
