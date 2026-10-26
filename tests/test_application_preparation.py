import pytest

from job_agent.apply.answer_bank import AnswerBank
from job_agent.apply.preparation import (
    approve_preparation, generate_preparation, get_preparation, is_approved,
    update_preparation,
)
from job_agent.tailor.career_facts import CareerFacts, Project


JOB = {"id": "j1", "title": "ML Intern", "company": "Example AI",
       "verification": {"status": "verified"},
       "eligibility": {"status": "eligible"},
       "description": "Build Python machine learning systems."}
FACTS = CareerFacts(projects=(Project(name="Classifier", description="Text classifier",
    technologies=("Python", "scikit-learn"), evidence=("resume project section",)),),
    skills_inventory={"languages": ("Python",)})


class Settings:
    gemini_api_key = ""


def test_generation_is_job_gated_and_missing_evidence_requires_review(tmp_path):
    path = tmp_path / "application_preparations.json"
    prep = generate_preparation(path=path, job=JOB, facts=FACTS,
        bank=AnswerBank(), resume_status="READY", settings=Settings(), jd=JOB["description"])
    assert prep["status"] == "NEEDS_REVIEW"
    assert any(a["key"] == "graduation" and a["status"] == "NEEDS_REVIEW"
               for a in prep["answers"])
    skills = next(a for a in prep["answers"] if a["key"] == "technical_skills")
    assert skills["answer"].lower() == "python"
    assert any("Python" in item for item in skills["evidence"])
    with pytest.raises(ValueError, match="verified and eligible"):
        generate_preparation(path=path, job={**JOB, "eligibility":{"status":"review"}},
            facts=FACTS, bank=AnswerBank(), resume_status="READY", settings=Settings(), jd="")
    with pytest.raises(ValueError, match="validated tailored resume"):
        generate_preparation(path=path, job=JOB, facts=FACTS, bank=AnswerBank(),
            resume_status="NEEDS_REVIEW", settings=Settings(), jd="")


def test_approval_requires_all_answers_reviewed_and_changes_invalidate(tmp_path):
    path = tmp_path / "application_preparations.json"
    prep = generate_preparation(path=path, job=JOB, facts=FACTS,
        bank=AnswerBank(), resume_status="READY", settings=Settings(), jd=JOB["description"])
    answers = {a["key"]: "Reviewed truthful response." for a in prep["answers"]}
    edited = update_preparation(path, "j1", {"answers": answers,
        "recruiter_message":"Hello, I am interested.", "cover_note":"Python project note."})
    assert edited["status"] == "ANSWERS_READY"
    approved = approve_preparation(path, "j1")
    assert is_approved(approved, job=JOB)
    update_preparation(path, "j1", {"answers": {"fit": "Changed after approval."}})
    assert not is_approved(get_preparation(path, "j1"), job=JOB)


def test_fingerprint_ties_approval_to_job_and_resume(tmp_path):
    path = tmp_path / "application_preparations.json"
    resume = tmp_path / "resume.pdf"; resume.write_bytes(b"validated pdf")
    prep = generate_preparation(path=path, job=JOB, facts=FACTS, bank=AnswerBank(),
        resume_status="READY", settings=Settings(), jd=JOB["description"], resume_path=resume)
    update_preparation(path, "j1", {"answers": {a["key"]: "Reviewed." for a in prep["answers"]}})
    approved = approve_preparation(path, "j1")
    assert is_approved(approved, job=JOB, resume_path=resume)
    resume.write_bytes(b"changed pdf")
    assert not is_approved(approved, job=JOB, resume_path=resume)


def test_unsupported_claim_is_flagged_by_existing_truthfulness_gate(tmp_path):
    path = tmp_path / "application_preparations.json"
    prep = generate_preparation(path=path, job=JOB, facts=FACTS,
        bank=AnswerBank(), resume_status="READY", settings=Settings(), jd=JOB["description"])
    answer = {a["key"]: "Reviewed truthful response." for a in prep["answers"]}
    answer["fit"] = "I worked at Invented Corporation and improved output by 80%."
    result = update_preparation(path, "j1", {"answers": answer}, facts=FACTS,
                                company=JOB["company"])
    assert result["content_issues"]
    assert result["status"] == "NEEDS_REVIEW"
    with pytest.raises(ValueError, match="Resolve every NEEDS_REVIEW"):
        approve_preparation(path, "j1")
