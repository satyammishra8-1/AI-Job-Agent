from job_agent.matching import (
    assess_eligibility, evidence_matches, extract_requirements, match_decision,
)
from job_agent.models import Job
from job_agent.tailor.career_facts import CareerFacts, Employer, Project


def job(title="Software Engineering Intern", description="",
        location="Bengaluru, India", country="India", remote=False):
    return Job(id="j1", title=title, company="Acme", location=location,
               url="https://example.com/job", source="greenhouse",
               country=country, remote=remote, description=description)


def profile(**kwargs):
    return CareerFacts(location="Bengaluru, India", **kwargs)


def test_student_is_eligible_for_internship():
    result = assess_eligibility(job(), profile(career_status="student"))
    assert result.status == "eligible"


def test_fresher_is_eligible_for_graduate_role():
    result = assess_eligibility(
        job("Graduate Software Engineer"), profile(career_status="fresher"))
    assert result.status == "eligible"


def test_hard_experience_mismatch_overrides_fit():
    role = job("Software Engineer", "Must have 3+ years professional experience.")
    result = assess_eligibility(role, profile(career_status="fresher"))
    assert result.status == "not_eligible"
    assert match_decision(result, 95) == "skip"


def test_unknown_eligibility_requirements_go_to_review():
    role = job("Software Engineer", "Bachelor's degree required.")
    result = assess_eligibility(role, profile(career_status="experienced"))
    assert result.status == "review"
    assert any("education" in reason.lower() for reason in result.reasons)


def test_missing_required_skill_evidence_goes_to_review_not_rejection():
    role = job("Python Developer", "Python required.")
    result = assess_eligibility(role, profile(career_status="experienced"))
    assert result.status == "review"
    assert any("Python" in reason for reason in result.reasons)


def test_graduation_year_hard_mismatch_is_not_eligible():
    role = job("Graduate Software Engineer",
               "Must graduate in 2027 to qualify.")
    result = assess_eligibility(role, profile(career_status="fresher",
                                               expected_graduation=2028))
    assert result.status == "not_eligible"


def test_work_authorization_uses_explicit_answer_bank_facts():
    role = job("Software Engineer", "Must be authorized to work in India.")
    unknown = assess_eligibility(role, profile(career_status="experienced"))
    supported = assess_eligibility(
        role, profile(career_status="experienced"),
        authorization={"authorized_countries": ["India"]})
    assert unknown.status == "review"
    assert supported.status == "eligible"


def test_50_to_64_fit_can_be_recommended_when_eligible():
    eligibility = assess_eligibility(job(), profile(career_status="student"))
    assert match_decision(eligibility, 55) == "recommended"


def test_requirement_extraction_separates_required_and_preferred_skills():
    requirements = extract_requirements(job(
        "Python Engineer", "Python required. AWS preferred. Must have 2 years experience."))
    assert "Python" in requirements.required_skills
    assert "AWS" in requirements.preferred_skills
    assert requirements.required_experience_years == 2


def test_explanation_uses_only_profile_skills_with_evidence():
    facts = profile(
        career_status="experienced",
        skills_inventory={"Languages": ("Python", "React")},
        employers=(Employer(company="Acme", title="Developer", duration="2024-present",
                            real_skills=("Python",)),),
        projects=(Project(name="Unverified", technologies=("React",)),),
    )
    role = job("Python Developer", "Python required. React required. AWS preferred.")
    matches, gaps = evidence_matches(role, facts)
    assert matches == ("Python — evidence: Acme",)
    assert gaps == ("React",)
