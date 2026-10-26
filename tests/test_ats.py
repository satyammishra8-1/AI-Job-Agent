from job_agent.ats import ats_readiness
from job_agent.models import Job
from job_agent.tailor.career_facts import CareerFacts, Employer


def _job(description: str) -> Job:
    return Job(id="j1", title="Developer", company="Acme",
               location="India", url="https://example.com", source="greenhouse",
               description=description)


def test_keyword_coverage_distinguishes_evidence_partial_and_missing():
    facts = CareerFacts(
        skills_inventory={"Languages": ("Python", "React")},
        employers=(Employer(company="Acme", title="Developer", duration="2024-present",
                            real_skills=("Python",)),),
    )
    result = ats_readiness(_job("Python required. React required. AWS required."), facts)
    assert result["readiness"] == 50
    assert (result["matched"], result["partial"], result["missing"]) == (1, 1, 1)
    assert [(item["keyword"], item["status"]) for item in result["keywords"]] == [
        ("Python", "MATCHED"), ("React", "PARTIAL"), ("AWS", "MISSING")]


def test_keyword_evidence_names_its_actual_source():
    facts = CareerFacts(skill_evidence={"PyTorch": ("Project: classifier",)})
    result = ats_readiness(_job("PyTorch experience required."), facts)
    assert result["keywords"][0]["category"] == "AI and machine learning"
    assert result["keywords"][0]["evidence"] == "Project: classifier"


def test_no_curated_keywords_means_no_claimed_score():
    result = ats_readiness(_job("Work with a curious, collaborative team."), CareerFacts())
    assert result["readiness"] is None
    assert result["total"] == 0


def test_role_and_qualification_keywords_use_profile_facts():
    facts = CareerFacts(role="Software Engineer", career_status="student",
                        internship_preference=True)
    job = _job("Software Engineer internship for an entry-level candidate.")
    result = ats_readiness(job, facts)
    by_keyword = {item["keyword"]: item for item in result["keywords"]}
    assert by_keyword["Software Engineer"]["category"] == "Role terminology"
    assert by_keyword["Software Engineer"]["status"] == "MATCHED"
    assert by_keyword["Internship"]["category"] == "Qualifications"
    assert by_keyword["Internship"]["status"] == "MATCHED"
    assert by_keyword["Entry-level"]["status"] == "MATCHED"
