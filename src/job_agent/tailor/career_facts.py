"""Immutable career facts — the source of truth the tailoring engine obeys.

Loaded from ``career_facts.yaml`` (real: gitignored; demo: committed fake). The
frozen models make employers/titles/durations/certs read-only in memory, and the
helper accessors give ``verify.py`` the exact allow-lists it enforces.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from job_agent.tailor.textnorm import norm as _norm


class Certification(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    issuer: str | None = None
    year: str | None = None
    evidence: tuple[str, ...] = ()


class EducationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    degree: str
    field: str = ""
    college: str = ""
    graduation_year: int | None = None
    cgpa: str | None = None
    evidence: tuple[str, ...] = ()


class Project(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: str = ""
    technologies: tuple[str, ...] = ()
    role: str = ""
    evidence: tuple[str, ...] = ()
    github_url: str | None = None

    def has_evidence(self) -> bool:
        return bool(self.name.strip()) and any(source.strip() for source in self.evidence)


class Employer(BaseModel):
    model_config = ConfigDict(frozen=True)
    company: str
    title: str
    location: str = ""
    duration: str
    project_description: str = ""
    real_bullets: tuple[str, ...] = ()
    real_metrics: tuple[str, ...] = ()
    real_skills: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


class CareerFacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Empty defaults allow the dashboard to initialize an editable profile
    # without inventing personal data. Tailoring still requires usable facts.
    name: str = ""
    role: str = ""
    email: str = ""
    phone: str = ""
    location: str | None = None
    links: tuple[str, ...] = ()
    education: tuple[str, ...] = ()
    certifications: tuple[Certification, ...] = ()
    skills_inventory: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # Optional editable master-profile fields. Existing career facts remain
    # valid; claims without evidence are not treated as verified matching facts.
    skill_evidence: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    education_details: tuple[EducationRecord, ...] = ()
    career_status: Literal["student", "fresher", "experienced"] | None = None
    current_year: str | None = None
    expected_graduation: int | None = None
    experience_years: float | None = Field(default=None, ge=0)
    internship_preference: bool | None = None
    preferred_locations: tuple[str, ...] = ()
    remote_preference: Literal["remote", "hybrid", "onsite", "any"] | None = None
    projects: tuple[Project, ...] = ()
    employers: tuple[Employer, ...] = ()

    # --- allow-lists consumed by the no-drift verifier ---------------------

    def employer_companies(self) -> set[str]:
        return {_norm(e.company) for e in self.employers}

    def employer_identities(self) -> set[tuple[str, str, str]]:
        """The (company, title, duration) triples that must survive unchanged."""
        return {(_norm(e.company), _norm(e.title), _norm(e.duration)) for e in self.employers}

    def certification_names(self) -> set[str]:
        return {_norm(c.name) for c in self.certifications}

    def real_skills(self) -> set[str]:
        """Skills supported by a listed role or evidence-backed profile item."""
        skills = {_norm(s) for group in self.skills_inventory.values() for s in group}
        for e in self.employers:
            skills |= {_norm(s) for s in e.real_skills}
        for project in self.projects:
            if project.has_evidence():
                skills |= {_norm(s) for s in project.technologies}
        skills |= {_norm(s) for s, sources in self.skill_evidence.items() if sources}
        # A named technical credential supports the technology stated in its name.
        skills |= {_norm(token) for cert in self.certifications for token in cert.name.split()}
        return skills

    def profile_configured(self) -> bool:
        """Whether any user-provided professional facts/preferences exist."""
        return bool(self.name or self.role or self.email or self.phone or self.location
                    or self.career_status or self.current_year or self.expected_graduation
                    or self.experience_years is not None or self.internship_preference is not None
                    or self.remote_preference or self.skills_inventory or self.skill_evidence
                    or self.links
                    or self.education or self.education_details or self.projects
                    or self.employers or self.certifications
                    or self.preferred_locations)


def load_career_facts(path: str | Path) -> CareerFacts:
    """Load and validate a career-facts YAML file with a readable error on failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Career facts not found: {path}. Generate it from the base resume first "
            f"(job_agent.tailor.extract)."
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse {path} as YAML: {exc}") from exc
    try:
        return CareerFacts.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid career facts {path}:\n{exc}") from exc
