"""Conservative job provenance and risk verification."""

from __future__ import annotations

from datetime import datetime, timezone

from job_agent.config import LocationRule, SearchProfile, SourceRef
from job_agent.models import Job, ScoredJob
from job_agent.scoring import score_one
from job_agent.search import run
from job_agent.seen_cache import SeenCache
from job_agent.sources.base import JobSource
from job_agent.verification import verify_job


def make_job(**updates) -> Job:
    data = {
        "id": "job-1", "title": "Software Engineer Intern", "company": "Acme",
        "location": "Bengaluru, India", "url": "https://acme.com/careers/1",
        "apply_url": "https://boards.greenhouse.io/embed/job_app?for=acme&token=1",
        "source": "greenhouse", "description": "Build software with Python.",
    }
    data.update(updates)
    return Job(**data)


def test_verified_employer_ats_posting_is_verified():
    result = verify_job(make_job())
    assert result.status == "verified"
    assert result.evidence.canonical_url == "https://acme.com/careers/1"
    assert result.evidence.application_domain == "boards.greenhouse.io"
    assert "employer_ats_posting" in {signal.code for signal in result.signals}


def test_cross_company_feed_without_canonical_posting_needs_review():
    result = verify_job(make_job(source="remoteok", url="https://remoteok.com/jobs/1",
                                 apply_url="https://acme.com/careers/1"))
    assert result.status == "needs_review"
    assert result.evidence.canonical_url is None
    assert "canonical_posting_unconfirmed" in {signal.code for signal in result.signals}


def test_missing_posting_url_needs_review_even_for_company_ats_source():
    result = verify_job(make_job(url=""))
    assert result.status == "needs_review"
    assert result.evidence.canonical_url is None


def test_suspicious_application_shortener_is_flagged():
    result = verify_job(make_job(apply_url="https://bit.ly/apply-now"))
    assert result.status == "needs_review"
    assert "hidden_application_destination" in {signal.code for signal in result.signals}


def test_private_application_domain_is_suspicious():
    result = verify_job(make_job(apply_url="http://127.0.0.1/apply"))
    assert result.status == "suspicious"
    assert "unsafe_application_domain" in {signal.code for signal in result.signals}


def test_explicit_pay_to_apply_language_is_suspicious():
    result = verify_job(make_job(description="Candidates must pay to apply."))
    assert result.status == "suspicious"
    assert "upfront_payment_request" in {signal.code for signal in result.signals}


def test_early_sensitive_data_request_is_suspicious():
    result = verify_job(make_job(
        description="Send your bank account details before the interview to proceed."))
    assert result.status == "suspicious"
    assert "sensitive_data_request" in {signal.code for signal in result.signals}


def test_guaranteed_job_with_no_screening_is_suspicious_language():
    result = verify_job(make_job(description=(
        "Guaranteed job placement with no interview or experience required.")))
    assert result.status == "suspicious"
    assert "unrealistic_job_promise" in {signal.code for signal in result.signals}


def test_application_domain_mismatch_needs_review_not_a_scam_claim():
    result = verify_job(make_job(apply_url="https://unrelated.example/apply"))
    assert result.status == "needs_review"
    assert "application_domain_mismatch" in {signal.code for signal in result.signals}


def test_posting_domain_mismatch_needs_review():
    result = verify_job(make_job(url="https://unrelated.example/jobs/1"))
    assert result.status == "needs_review"
    assert "employer_domain_mismatch" in {signal.code for signal in result.signals}


def test_safe_ordinary_description_does_not_add_risk_signals():
    result = verify_job(make_job(description=(
        "We offer competitive compensation and support professional development.")))
    assert result.status == "verified"
    assert not any(signal.severity == "risk" for signal in result.signals)


def test_multiple_independent_risk_signals_are_preserved():
    result = verify_job(make_job(
        apply_url="https://bit.ly/apply-now",
        description="Pay to apply and send your bank account details before the interview."))
    assert result.status == "suspicious"
    assert {signal.code for signal in result.signals} >= {
        "hidden_application_destination", "upfront_payment_request",
        "sensitive_data_request",
    }


def test_verification_runs_after_discovery_enrichment_before_scoring(tmp_path):
    job = make_job()

    class Source(JobSource):
        ats = "greenhouse"

        def fetch(self):
            return [job]

    profile = SearchProfile(
        keywords=["software engineer"],
        location=LocationRule(remote_ok=True, allowed_countries=["IN"]),
        sources=[SourceRef(ats="greenhouse", board="acme")],
        candidate_summary="software engineering candidate",
    )
    outcome = run(profile, seen_cache=SeenCache(tmp_path / "seen.json"),
                  now=datetime(2026, 10, 1, tzinfo=timezone.utc),
                  source_factory=lambda *_: Source("acme"))
    assert outcome.jobs[0].verification.status == "verified"


def test_verification_survives_existing_search_json_persistence(tmp_path):
    from job_agent.store import load_job_record, save_search

    job = make_job().model_copy(update={"verification": verify_job(make_job())})
    path = tmp_path / "last_search.json"
    save_search([ScoredJob(job=job, score=82, verdict="strong")],
                ["greenhouse/acme"], path)
    saved = load_job_record(path, job.id)
    assert saved["verification"]["status"] == "verified"
    assert saved["verification"]["evidence"]["canonical_url"] == job.url


def test_gemini_scoring_contract_remains_unchanged_with_verified_job():
    class Response:
        text = ('{"score": 82, "verdict": "strong", "reasons": ["Python"], '
                '"missing_requirements": []}')

    class Models:
        @staticmethod
        def generate_content(**kwargs):
            return Response()

    class Client:
        models = Models()

    from job_agent.config import SearchProfile

    profile = SearchProfile.model_validate({
        "keywords": ["software"], "location": {"remote_ok": True},
        "sources": [{"ats": "greenhouse", "board": "acme"}],
        "candidate_summary": "software engineer",
    })
    job = make_job().model_copy(update={"verification": verify_job(make_job())})
    result = score_one(Client(), "test-model", job, profile)
    assert isinstance(result, ScoredJob)
    assert (result.score, result.verdict) == (82, "strong")


def test_verified_job_ranks_ahead_of_stronger_unverified_fit():
    verified = make_job().model_copy(update={"verification": verify_job(make_job())})
    review = make_job(id="job-2", source="remoteok", url="https://remoteok.com/jobs/2",
                      apply_url="https://acme.com/apply")
    review = review.model_copy(update={"verification": verify_job(review)})
    assert ScoredJob(job=verified, score=60, verdict="possible").sort_key > \
           ScoredJob(job=review, score=99, verdict="strong").sort_key


def test_apply_runner_blocks_suspicious_jobs_before_recording_or_opening(tmp_path):
    import pytest

    from job_agent.apply.answer_bank import AnswerBank, Contact
    from job_agent.apply.runner import ApplyConfig, run_apply

    log = tmp_path / "applications.json"
    cfg = ApplyConfig(
        apply_url="https://bit.ly/apply", bank=AnswerBank.model_validate({}),
        contact=Contact(name="", email="", phone=""),
        verification_status="suspicious", applications_log=log,
    )
    with pytest.raises(ValueError, match="explicit human review"):
        run_apply(cfg)
    assert not log.exists()
