"""Deterministic profile analysis: completeness and evidence status, without inference."""
from __future__ import annotations

from job_agent.matching import evidenced_profile_skills
from job_agent.tailor.career_facts import CareerFacts


def analyze_profile(facts: CareerFacts) -> dict:
    """Return actionable completeness and a transparent profile summary."""
    skills = evidenced_profile_skills(facts)
    claimed = {s.lower(): s for group in facts.skills_inventory.values() for s in group}
    claimed.update({s.lower(): s for s in facts.skill_evidence})
    for project in facts.projects:
        for skill in (*project.technologies, *project.relevant_skills):
            claimed.setdefault(skill.lower(), skill)
    for employer in facts.employers:
        for skill in employer.real_skills:
            claimed.setdefault(skill.lower(), skill)
    missing: list[str] = []
    categories = {
        "basic_information": bool(facts.name and facts.email and facts.location),
        "education": bool(facts.education or facts.education_details),
        "skills": bool(claimed),
        "projects": bool(facts.projects),
        "github": bool(facts.github_profile_url or any(p.github_url for p in facts.projects)),
        "experience": bool(facts.employers),
        "career_preferences": bool(facts.target_roles or facts.role or facts.career_direction
                                    or facts.preferred_locations or facts.remote_preference),
        "availability": bool(facts.availability or facts.internship_duration
                              or facts.preferred_start_date),
    }
    labels = {"basic_information": "Basic information (name, email, location)",
              "education": "Education", "skills": "Skills", "projects": "Projects",
              "github": "GitHub profile or project links", "experience": "Experience",
              "career_preferences": "Career preferences and target roles",
              "availability": "Availability and internship duration"}
    if not facts.name: missing.append("Name")
    if not facts.email: missing.append("Email")
    if not facts.location: missing.append("Current location")
    if not categories["education"]: missing.append(labels["education"])
    if not categories["skills"]: missing.append(labels["skills"])
    if not categories["projects"]: missing.append(labels["projects"])
    if not categories["github"]: missing.append(labels["github"])
    if not categories["experience"]: missing.append(labels["experience"])
    if not categories["career_preferences"]: missing.append("Target role or career direction")
    if not categories["availability"]:
        missing.extend(["Availability", "Preferred internship duration"])
    unsupported = sorted(value for key, value in claimed.items() if key not in skills)
    if unsupported:
        missing.append(f"Evidence for {len(unsupported)} claimed skill(s): " + ", ".join(unsupported))
    percent = round(100 * sum(categories.values()) / len(categories))
    evidenced_projects = [p for p in facts.projects if p.has_evidence()]
    strongest_projects = [p.name for p in evidenced_projects]
    strongest = sorted((entry[0] for entry in skills.values()), key=str.casefold)
    return {
        "profile_id": facts.profile_id or None,
        "profile_version": facts.profile_version,
        "completeness": percent,
        "categories": categories,
        "missing": missing,
        "experience_level": facts.career_status or ("experienced" if facts.employers else "unknown"),
        "strongest_skills": strongest[:8],
        "supporting_skills": strongest[8:],
        "claimed_skills": sorted(claimed.values(), key=str.casefold),
        "evidence_backed_skills": strongest,
        "skill_evidence": [{"skill": name, "source": source}
                           for name, source in skills.values()],
        "unknown_or_unsupported_skills": unsupported,
        "target_roles": list(facts.target_roles) or ([facts.role] if facts.role else []),
        "target_locations": list(facts.preferred_locations),
        "preferred_work_mode": facts.remote_preference or "unknown",
        "strongest_projects": strongest_projects[:5],
        "career_direction": facts.career_direction or "unknown",
        "profile_strength": "strong" if percent >= 80 and len(skills) >= 3 else "developing" if percent >= 40 else "incomplete",
    }
