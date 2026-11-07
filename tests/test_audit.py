from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from job_agent.audit import AuditCollector, ReadOnlySeenCache, build_report
from job_agent.config import LocationRule, SearchProfile, SourceRef
from job_agent.models import Job, ScoredJob, EligibilityResult, VerificationResult
from job_agent.search import run
from job_agent.sources.base import JobSource


class AuditSource(JobSource):
    ats = "audit-fake"

    def __init__(self, board, jobs):
        super().__init__(board)
        self.jobs = jobs

    def fetch(self):
        return self.jobs


def _profile():
    return SearchProfile(
        keywords=["data engineer"],
        location=LocationRule(allowed_countries=["India"], remote_ok=True),
        sources=[SourceRef(ats="audit-fake", board="q")],
        max_seniority="senior",
    )


def _job(job_id, title, **extra):
    data = dict(id=job_id, title=title, company=job_id, location="Bengaluru, India",
                country="India", url="https://example.com/job", source="demo",
                posted_at=datetime.now(timezone.utc) - timedelta(hours=1))
    data.update(extra)
    return Job(**data)


def test_audit_pipeline_reports_stages_without_saving_seen_cache(tmp_path):
    seen_path = tmp_path / "seen.json"
    original = {"demo:old": "2026-11-01T00:00:00+00:00"}
    seen_path.write_text(json.dumps(original))
    before = seen_path.read_bytes()
    jobs = [
        _job("1", "Data Engineer Intern"),
        _job("2", "Frontend Engineer"),
        _job("3", "Staff Data Engineer"),
        _job("4", "Data Engineer Intern", company="1"),
    ]
    collector = AuditCollector()
    outcome = run(_profile(), seen_cache=ReadOnlySeenCache(seen_path),
                  now=datetime.now(timezone.utc),
                  source_factory=lambda ats, board: AuditSource(board, jobs),
                  audit=collector)

    assert outcome.counts.fetched == 4
    assert outcome.counts.after_keyword == 3
    assert outcome.counts.after_seniority == 2
    assert collector.stages["after_pagination"] == 4
    assert collector.stages["after_dedup"] == 1
    assert collector.rejections["keyword"] == 1
    assert collector.rejections["seniority"] == 1
    assert collector.rejections["deduplication"] == 1
    assert seen_path.read_bytes() == before


def test_audit_final_stages_count_verified_eligible_and_matched(tmp_path):
    collector = AuditCollector()
    job = _job("1", "Data Engineer Intern").model_copy(update={
        "verification": VerificationResult(status="verified", reason="official"),
    })
    outcome = type("Outcome", (), {
        "jobs": [job],
        "counts": type("Counts", (), {"after_experience": 1})(),
        "per_source": {"fake/q": 1},
        "warnings": [],
    })()
    scored = ScoredJob(job=job, score=88, verdict="strong",
                       eligibility=EligibilityResult(status="eligible"),
                       decision="recommended")
    report = build_report(outcome, collector, [scored])

    assert report["read_only"] is True
    assert report["stages"]["after_internship_entry_filter"] == 1
    assert report["stages"]["after_verification"] == 1
    assert report["stages"]["after_eligibility"] == 1
    assert report["stages"]["after_profile_matching"] == 1
    assert report["sources"]["fake/q"]["jobs_returned"] == 1


def test_audit_classifier_is_reporting_only_and_covers_training_titles():
    from job_agent.audit import is_internship_or_entry_level

    assert is_internship_or_entry_level("Graduate Engineer Trainee")
    assert is_internship_or_entry_level("Software Engineer Intern")
    assert not is_internship_or_entry_level("Senior Data Engineer")
