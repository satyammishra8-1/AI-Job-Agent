"""Scoring: defensive JSON parsing, retry-once, and the unscored fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

from job_agent.config import LocationRule, SearchProfile, Settings, SourceRef
from job_agent.models import Job
from job_agent.scoring import score_jobs, score_one

PROFILE = SearchProfile(
    keywords=["data engineer"],
    location=LocationRule(),
    sources=[SourceRef(ats="fake", board="b")],
    candidate_summary="data/ML engineer",
)
JOB = Job(id="1", title="Data Engineer", company="Acme", location="Remote",
          url="http://x", source="demo", remote=True, country="US",
          description="Build pipelines.")


class FakeModels:
    """Returns queued responses (or raises queued exceptions) per create()."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def text_response(payload: str):
    return SimpleNamespace(text=payload, function_calls=[])


def tool_response(data: dict):
    return SimpleNamespace(text=None, function_calls=[
        SimpleNamespace(name="record_fit", args=data)])


VALID = json.dumps({"score": 82, "verdict": "strong",
                    "reasons": ["good match"], "missing_requirements": ["Kafka"]})


def test_structured_valid_output():
    client = FakeClient([text_response(VALID)])
    result = score_one(client, "m", JOB, PROFILE, method="structured")
    assert result.score == 82
    assert result.verdict == "strong"
    assert result.reasons == ("good match",)
    assert result.missing_requirements == ("Kafka",)


def test_score_is_clamped_to_0_100():
    client = FakeClient([text_response(json.dumps(
        {"score": 150, "verdict": "strong", "reasons": [], "missing_requirements": []}))])
    result = score_one(client, "m", JOB, PROFILE)
    assert result.score == 100


def test_retries_once_then_succeeds():
    client = FakeClient([text_response("not json at all"), text_response(VALID)])
    result = score_one(client, "m", JOB, PROFILE)
    assert client.models.calls == 2
    assert result.verdict == "strong"


def test_unscored_after_two_failures():
    client = FakeClient([text_response("garbage"), text_response("still garbage")])
    result = score_one(client, "m", JOB, PROFILE)
    assert client.models.calls == 2
    assert result.verdict == "unscored"
    assert result.score is None
    assert result.job == JOB  # the job is preserved, not dropped


def test_tool_use_method_reads_tool_input():
    client = FakeClient([tool_response(
        {"score": 70, "verdict": "possible", "reasons": ["ok"], "missing_requirements": []})])
    result = score_one(client, "m", JOB, PROFILE, method="tool")
    assert result.verdict == "possible"
    assert result.score == 70


def test_score_jobs_uses_injected_client():
    settings = Settings(gemini_api_key="x", model="m")
    client = FakeClient([text_response(VALID), text_response(VALID)])
    results = score_jobs([JOB, JOB], settings, PROFILE, client=client)
    assert len(results) == 2
    assert all(r.verdict == "strong" for r in results)


def test_score_jobs_computes_eligibility_and_recommends_55_percent_fit():
    from job_agent.tailor.career_facts import CareerFacts

    job = JOB.model_copy(update={"title": "Software Engineering Intern",
                                 "location": "Bengaluru, India", "country": "India"})
    candidate = CareerFacts(career_status="student", location="Bengaluru, India")
    client = FakeClient([text_response(json.dumps(
        {"score": 55, "verdict": "possible", "reasons": [], "missing_requirements": []}))])
    result = score_jobs([job], Settings(gemini_api_key="x", model="m"), PROFILE,
                        client=client, candidate=candidate)[0]
    assert result.score == 55
    assert result.eligibility.status == "eligible"
    assert result.decision == "recommended"


def test_scoring_prompt_only_marks_evidenced_skills_as_supported():
    from job_agent.scoring import build_user_prompt
    from job_agent.tailor.career_facts import CareerFacts, Employer, Project

    candidate = CareerFacts(
        skills_inventory={"Languages": ("Python", "Java")},
        employers=(Employer(company="Acme", title="Developer", duration="2024-present",
                            real_skills=("SQL",)),),
        projects=(Project(name="Unlinked", technologies=("React",)),),
    )
    prompt = build_user_prompt(JOB, PROFILE, candidate)
    assert "SQL (evidence: Acme)" in prompt
    assert "Python (evidence:" not in prompt
    assert "React (evidence:" not in prompt
