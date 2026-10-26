"""Deterministic ATS keyword coverage grounded in the editable career profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from job_agent.matching import SKILL_TERMS, evidenced_profile_skills
from job_agent.models import Job
from job_agent.tailor.career_facts import CareerFacts


@dataclass(frozen=True)
class KeywordResult:
    keyword: str
    category: str
    status: str
    evidence: str | None = None


_CATEGORIES = {
    "Languages": {"python", "java", "javascript", "typescript", "c++", "c#", "go", "rust"},
    "Frameworks and libraries": {"react", "angular", "vue", "django", "fastapi", "flask", "spring"},
    "Databases": {"sql", "postgresql", "mysql", "mongodb", "redis"},
    "Cloud and infrastructure": {"aws", "azure", "gcp", "docker", "kubernetes", "linux"},
    "Tools": {"git"},
    "AI and machine learning": {"machine learning", "deep learning", "tensorflow", "pytorch", "nlp"},
    "Software engineering concepts": {"data structures", "algorithms", "rest api", "graphql", "microservices", "html", "css"},
    "Role terminology": {"software engineer", "software engineering", "sde", "backend", "backend engineering",
                         "frontend", "frontend engineering", "full stack", "machine learning engineer",
                         "ai engineer", "artificial intelligence", "data engineer", "data engineering",
                         "devops", "site reliability engineer", "research engineer"},
    "Qualifications": {"bachelor's degree", "bachelor degree", "master's degree", "master degree",
                       "phd", "internship", "entry-level", "fresher"},
}
_ROLE_TERMS = (
    "Software Engineer", "Software Engineering", "SDE", "Backend", "Backend Engineering",
    "Frontend", "Frontend Engineering", "Full Stack", "Machine Learning Engineer",
    "AI Engineer", "Artificial Intelligence", "Data Engineer", "Data Engineering",
    "DevOps", "Site Reliability Engineer", "Research Engineer",
)
_QUALIFICATION_TERMS = (
    "Bachelor's degree", "Bachelor degree", "Master's degree", "Master degree",
    "PhD", "Internship", "Entry-level", "Fresher",
)


def _category(keyword: str) -> str:
    low = keyword.lower()
    return next((name for name, values in _CATEGORIES.items() if low in values), "Role terminology")


def _mentioned(text: str, term: str) -> bool:
    # Keep punctuation-bearing names (C++, Node.js) usable while avoiding
    # accidental matches inside longer words.
    import re

    pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text.lower()))


def ats_readiness(job: Job, facts: CareerFacts) -> dict:
    """Return explainable keyword coverage; this is not a commercial ATS score.

    Only the application's curated requirement vocabulary is measured. A fact
    without evidence can be PARTIAL, but it is never treated as a resume claim.
    """
    text = f"{job.title}\n{job.description or ''}"
    keywords = [term for term in (*SKILL_TERMS, *_ROLE_TERMS, *_QUALIFICATION_TERMS)
                if _mentioned(text, term)]
    profile = evidenced_profile_skills(facts)
    listed = {skill.lower(): skill for group in facts.skills_inventory.values() for skill in group}
    listed.update({term.lower(): term for e in facts.employers for term in e.real_skills})
    results = []
    for term in keywords:
        key = term.lower()
        if key in profile:
            status, evidence = "MATCHED", profile[key][1]
        elif key in listed:
            status, evidence = "PARTIAL", "Listed in profile without evidence"
        elif term in _ROLE_TERMS:
            roles = [facts.role, *(employer.title for employer in facts.employers),
                     *(project.role for project in facts.projects)]
            source = next((role for role in roles if role and _mentioned(role, term)), None)
            status, evidence = (("MATCHED", f"Profile role: {source}") if source
                                else ("MISSING", None))
        elif term in _QUALIFICATION_TERMS:
            education = " ".join([*facts.education,
                                  *(f"{item.degree} {item.field} {item.college}"
                                    for item in facts.education_details)])
            status, evidence = "MISSING", None
            if "degree" in term and _mentioned(education, term.replace("'s", "")):
                status, evidence = "MATCHED", "Education in profile"
            elif term.lower() in {"internship", "entry-level", "fresher"} and (
                (term.lower() == "internship" and facts.internship_preference is True)
                or (term.lower() in {"entry-level", "fresher"}
                    and facts.career_status in {"student", "fresher"})
            ):
                status, evidence = "MATCHED", "Career preferences/status in profile"
        else:
            status, evidence = "MISSING", None
        results.append(KeywordResult(term, _category(term), status, evidence))
    total = len(results)
    matched = sum(r.status == "MATCHED" for r in results)
    partial = sum(r.status == "PARTIAL" for r in results)
    readiness = round(100 * (matched + 0.5 * partial) / total) if total else None
    return {
        "readiness": readiness,
        "total": total,
        "matched": matched,
        "partial": partial,
        "missing": total - matched - partial,
        "method": "matched=1, partial=0.5, missing=0; rounded to a whole percent",
        "keywords": [asdict(r) for r in results],
    }
