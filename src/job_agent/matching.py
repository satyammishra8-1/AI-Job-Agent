"""Conservative profile-based eligibility and evidence-grounded explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from job_agent.models import EligibilityResult, Job, MatchDecision
from job_agent.tailor.career_facts import CareerFacts


@dataclass(frozen=True)
class JobRequirements:
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    required_experience_years: int | None = None
    location_text: tuple[str, ...] = ()


# A deliberately small vocabulary: extraction is a transparent text check, not
# an LLM call. Extend only when supported by current profile/job data.
SKILL_TERMS = (
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "React", "Angular", "Vue", "Node.js", "Django", "FastAPI", "Flask",
    "Spring", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "AWS",
    "Azure", "GCP", "Docker", "Kubernetes", "Linux", "Git", "Machine Learning",
    "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Data Structures",
    "Algorithms", "HTML", "CSS", "REST API", "GraphQL", "Microservices",
)


def _has(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text.lower()))


def _required_experience_years(text: str) -> int | None:
    """Extract hard years requirements without borrowing cues across sentences."""
    hard = []
    for sentence in re.split(r"[.;\n]+", text or ""):
        low = sentence.lower()
        if any(cue in low for cue in ("preferred", "ideally", "nice to have", "bonus", "or equivalent")):
            continue
        if not any(cue in low for cue in ("required", "require", "must", "minimum", "at least")):
            continue
        hard.extend(int(m.group(1)) for m in
                    re.finditer(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?years?", low))
    return max(hard) if hard else None


def extract_requirements(job: Job) -> JobRequirements:
    text = f"{job.title}\n{job.description or ''}"
    required, preferred = [], []
    sentences = re.split(r"[.;\n]+", text)
    for skill in SKILL_TERMS:
        if not _has(text, skill):
            continue
        # A nearby preferred cue keeps a skill out of required requirements.
        contexts = [sentence.lower() for sentence in sentences if _has(sentence, skill)]
        if contexts and all(any(cue in context for cue in
                                ("preferred", "nice to have", "bonus", "plus"))
                            for context in contexts):
            preferred.append(skill)
        elif (skill.lower() in job.title.lower()
              or any(any(cue in context for cue in
                         ("required", "must", "proficient", "strong knowledge",
                          "experience with", "experience in")) for context in contexts)):
            required.append(skill)
    locations = tuple(dict.fromkeys(
        term for term in ("Bengaluru", "Bangalore", "India", "Remote", "Hybrid")
        if _has(f"{job.location} {job.description}", term)))
    return JobRequirements(tuple(required), tuple(preferred),
                           _required_experience_years(job.description or ""), locations)


def evidenced_profile_skills(facts: CareerFacts) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    evidenced: dict[str, tuple[str, ...]] = {}
    for employer in facts.employers:
        for skill in employer.real_skills:
            evidenced.setdefault(skill.lower(), (skill, employer.company))
    for project in facts.projects:
        if project.has_evidence():
            for skill in project.technologies:
                evidenced.setdefault(skill.lower(), (skill, project.name))
    for skill, sources in facts.skill_evidence.items():
        if sources:
            evidenced.setdefault(skill.lower(), (skill, sources[0]))
    for skills in facts.skills_inventory.values():
        for skill in skills:
            if skill.lower() in evidenced:
                result.setdefault(skill.lower(), evidenced[skill.lower()])
    for cert in facts.certifications:
        for skill in SKILL_TERMS:
            if _has(cert.name, skill):
                evidenced.setdefault(skill.lower(), (skill, f"Certification: {cert.name}"))
    result.update(evidenced)
    return result


def evidence_matches(job: Job, facts: CareerFacts | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return only supported skill overlaps and explicit required skill gaps."""
    if facts is None:
        return (), ()
    requirements = extract_requirements(job)
    profile = evidenced_profile_skills(facts)
    matches = tuple(f"{profile[s.lower()][0]} — evidence: {profile[s.lower()][1]}"
                    for s in requirements.required_skills if s.lower() in profile)
    gaps = tuple(s for s in requirements.required_skills if s.lower() not in profile)
    return matches, gaps


def assess_eligibility(job: Job, facts: CareerFacts | None, *,
                       authorization: dict | None = None) -> EligibilityResult:
    """Reject only clear, hard mismatches; leave absent/ambiguous facts to review."""
    if facts is None:
        return EligibilityResult(status="review",
                                 reasons=("Master profile is not configured; eligibility cannot be assessed.",))

    reasons: list[str] = []
    unknown: list[str] = []
    title = job.title.lower()
    is_internship = bool(re.search(r"\bintern(ship)?\b", title))
    is_entry = bool(re.search(r"\b(fresher|entry.level|new.grad|graduate|junior)\b", title))
    status = (facts.career_status or "").lower()
    low_jd = (job.description or "").lower()

    if re.search(r"currently enrolled|must be enrolled|currently pursuing", low_jd):
        if status == "student":
            reasons.append("Required current-enrollment condition matches student status.")
        elif status in {"fresher", "experienced"}:
            return EligibilityResult(status="not_eligible", reasons=(
                "Role requires current enrollment; profile status is " + status + ".",))
        else:
            unknown.append("Role requires current enrollment, but student status is not recorded.")

    for sentence in re.split(r"[.;\n]+", low_jd):
        if not re.search(r"graduat(?:e|ing|ion)", sentence):
            continue
        if not any(cue in sentence for cue in ("required", "must", "by", "before", "between")):
            continue
        years = [int(year) for year in re.findall(r"\b20\d{2}\b", sentence)]
        if not years:
            continue
        if facts.expected_graduation is None:
            unknown.append("A hard graduation timing requirement is stated, but expected graduation is not recorded.")
        elif "before" in sentence and facts.expected_graduation >= min(years):
            return EligibilityResult(status="not_eligible", reasons=(
                "Expected graduation falls after the posting's required graduation cutoff.",))
        elif "by" in sentence and facts.expected_graduation > min(years):
            return EligibilityResult(status="not_eligible", reasons=(
                "Expected graduation falls after the posting's required graduation date.",))
        elif "between" in sentence and len(years) > 1 and not min(years) <= facts.expected_graduation <= max(years):
            return EligibilityResult(status="not_eligible", reasons=(
                "Expected graduation falls outside the posting's required graduation window.",))
        elif "in" in sentence and facts.expected_graduation not in years:
            return EligibilityResult(status="not_eligible", reasons=(
                "Expected graduation year does not match the posting's stated requirement.",))
        else:
            reasons.append("Expected graduation meets the stated timing requirement.")

    requirements = extract_requirements(job)
    years = requirements.required_experience_years
    if years is not None:
        if facts.experience_years is not None:
            if years > facts.experience_years:
                reasons.append(f"Role requires {years}+ years; profile records {facts.experience_years:g} years.")
                return EligibilityResult(status="not_eligible", reasons=tuple(reasons))
            reasons.append(f"Hard experience requirement ({years}+ years) is within the recorded experience.")
        elif status in {"student", "fresher"}:
            reasons.append(f"Role requires {years}+ years; profile status is {status}.")
            return EligibilityResult(status="not_eligible", reasons=tuple(reasons))
        else:
            unknown.append("Professional experience requirement cannot be compared with a recorded experience total.")

    required_skills = requirements.required_skills
    unsupported_requirements = [skill for skill in required_skills
                                if skill.lower() not in evidenced_profile_skills(facts)]
    if unsupported_requirements:
        unknown.append("No profile evidence is recorded for required skills: "
                       + ", ".join(unsupported_requirements) + ".")

    if is_internship:
        if status == "student":
            reasons.append("Internship role is consistent with student status.")
        elif facts.internship_preference is not None:
            reasons.append("Internship is " + ("preferred" if facts.internship_preference
                                                else "not preferred") + " in the profile.")
            unknown.append("Internship enrollment eligibility is not established by preference alone.")
        else:
            unknown.append("Internship enrollment eligibility is not specified in the profile.")
    elif is_entry and status in {"student", "fresher"}:
        reasons.append(f"Entry-level role is consistent with {status} status.")
    elif status in {"student", "fresher"}:
        unknown.append("Role seniority is not explicit for the recorded student/fresher status.")
    elif not status:
        unknown.append("Career status is not specified in the profile.")
    else:
        reasons.append(f"Career status is recorded as {status}.")

    education_text = " ".join(
        [*facts.education,
         *(f"{e.degree} {e.field} {e.college}" for e in facts.education_details)]
    ).lower()
    degree_terms = ("bachelor's degree", "bachelor’s degree", "bachelor degree",
                    "master's degree", "master’s degree", "master degree")
    hard_degree = any(term in low_jd and any(cue in low_jd for cue in ("required", "must have", "minimum"))
                      for term in degree_terms)
    if hard_degree:
        if education_text:
            reasons.append("Education is recorded in the profile; degree equivalence may need human review.")
            unknown.append("Specific degree equivalence is not assessed automatically.")
        else:
            unknown.append("Job states a degree requirement, but education is not recorded.")

    if re.search(r"work authorization|visa sponsorship|authorized to work|right to work", low_jd):
        auth = authorization or {}
        from job_agent.geo import infer_country

        country = (job.country or infer_country(job.location) or "").strip().lower()
        allowed = {str(c).strip().lower() for c in auth.get("authorized_countries", [])}
        if country == "in":
            country = "india"
        if country in {"us", "usa", "united states of america"}:
            country = "united states"
        allowed = {"united states" if c in {"us", "usa", "united states of america"} else c
                   for c in allowed}
        is_us = country in {"us", "usa", "united states", "united states of america"}
        authorized = bool(country and country in allowed) or (is_us and auth.get("authorized_us") is True)
        explicitly_unauthorized = bool(country and allowed and country not in allowed) or (
            is_us and auth.get("authorized_us") is False)
        if authorized:
            reasons.append(f"Work authorization is recorded for {job.country or 'the job country'}.")
        elif explicitly_unauthorized:
            return EligibilityResult(status="not_eligible", reasons=(
                f"Work authorization is not recorded for {job.country or 'the job country'}.",))
        else:
            unknown.append("Work authorization requirements need human review; no compatible fact is recorded.")
        no_sponsor = bool(re.search(r"no (?:visa )?sponsorship|will not sponsor|unable to sponsor", low_jd))
        if no_sponsor and auth.get("requires_sponsorship") is True:
            return EligibilityResult(status="not_eligible", reasons=tuple(reasons + [
                "The posting does not offer sponsorship, while the profile records sponsorship is required."]))
        if no_sponsor and auth.get("requires_sponsorship") is False:
            reasons.append("The posting does not offer sponsorship and none is recorded as required.")

    if job.country:
        country = job.country.strip().lower()
        if "india" in country or country == "in":
            reasons.append("Job country is India.")
        elif facts.location and "india" in facts.location.lower():
            reasons.append(f"Job country is {job.country}, which differs from the recorded India location.")
            return EligibilityResult(status="not_eligible", reasons=tuple(reasons))
    elif job.location:
        loc = job.location.lower()
        if any(place in loc for place in ("india", "bengaluru", "bangalore")):
            reasons.append("Job location is compatible with the recorded India preference.")
        elif job.remote is True and "india" in (facts.location or "").lower():
            unknown.append("Remote role's country eligibility is not stated.")
        else:
            unknown.append("Location compatibility cannot be determined from the posting.")

    preferred = [place.lower() for place in facts.preferred_locations]
    job_location = job.location.lower()
    if preferred and any(place in job_location for place in preferred):
        reasons.append("Job location matches a preferred location in the profile.")
    if facts.remote_preference == "remote" and job.remote is True:
        reasons.append("Job is remote, matching the profile preference.")

    if unknown:
        return EligibilityResult(status="review", reasons=tuple(reasons + unknown))
    return EligibilityResult(status="eligible", reasons=tuple(reasons) or ("No hard mismatch was found.",))


def match_decision(eligibility: EligibilityResult, score: int | None) -> MatchDecision:
    if eligibility.status == "not_eligible":
        return "skip"
    if eligibility.status == "review" or score is None:
        return "review"
    if score >= 50:
        return "recommended"
    if score >= 40:
        return "review"
    return "skip"
