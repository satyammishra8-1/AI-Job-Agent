"""Typed, immutable domain models.

Every ATS source normalizes its raw JSON into a :class:`Job`. The scorer wraps a
``Job`` into a :class:`ScoredJob`. Both are frozen (immutable) — code creates new
objects rather than mutating existing ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The ATS/board a job came from. The last three are cross-company search
# sources (keyword-driven), not per-company boards.
SourceName = Literal["greenhouse", "lever", "ashby", "smartrecruiters", "demo",
                     "sr-search", "remotive", "remoteok"]

# The scorer's verdict. "unscored" is a distinct state used when the LLM output
# could not be parsed even after one retry (see scoring.py).
Verdict = Literal["strong", "possible", "skip", "unscored"]
VerificationStatus = Literal["verified", "needs_review", "suspicious"]
EligibilityStatus = Literal["eligible", "review", "not_eligible"]
MatchDecision = Literal["recommended", "review", "skip"]


class EligibilityResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: EligibilityStatus = "review"
    reasons: tuple[str, ...] = ()


class VerificationSignal(BaseModel):
    """One evidence-based verification or risk observation."""

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Literal["info", "review", "risk"]
    message: str


class VerificationEvidence(BaseModel):
    """URLs and domains the deterministic verifier actually inspected."""

    model_config = ConfigDict(frozen=True)

    canonical_url: str | None = None
    source_url: str | None = None
    employer_domain: str | None = None
    application_domain: str | None = None


class VerificationResult(BaseModel):
    """Conservative status plus the evidence supporting it."""

    model_config = ConfigDict(frozen=True)

    status: VerificationStatus
    signals: tuple[VerificationSignal, ...] = ()
    evidence: VerificationEvidence = Field(default_factory=VerificationEvidence)
    reason: str


class Job(BaseModel):
    """A single job posting, normalized across all ATS sources."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable per-source id (source-scoped, not global).")
    title: str
    company: str
    location: str = Field(description="Human-readable location, e.g. 'Remote' or 'Austin, TX'.")
    url: str = Field(description="Public posting URL a human can open.")
    source: SourceName
    apply_url: str | None = None
    # Timezone-aware. None when the source exposes no post date and no cache hit
    # exists yet (see seen_cache.py for the selected-window fallback).
    posted_at: datetime | None = None
    remote: bool | None = None
    # Best-effort country marker for the location rule; None when unknown.
    country: str | None = None
    description: str = ""
    verification: VerificationResult | None = None

    def dedup_key(self) -> str:
        """Canonical identity used to drop duplicates across boards/re-runs."""
        return "|".join([self.company.strip().lower(), self.title.strip().lower(),
                          self.location.strip().lower()])


class ScoredJob(BaseModel):
    """A :class:`Job` plus the LLM's fit assessment."""

    model_config = ConfigDict(frozen=True)

    job: Job
    # 0-100 fit score; None when the job could not be scored (verdict == "unscored").
    score: int | None = Field(default=None, ge=0, le=100)
    verdict: Verdict = "unscored"
    reasons: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    eligibility: EligibilityResult = Field(default_factory=EligibilityResult)
    supported_matches: tuple[str, ...] = ()
    supported_gaps: tuple[str, ...] = ()
    decision: MatchDecision = "review"

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """Prefer verified, eligible opportunities, then fit verdict and score."""
        verification_rank = {
            "verified": 2, "needs_review": 1, "suspicious": 0,
        }.get(self.job.verification.status if self.job.verification else None, 1)
        verdict_rank = {"strong": 3, "possible": 2, "skip": 1, "unscored": 0}
        eligibility_rank = {"eligible": 2, "review": 1, "not_eligible": 0}
        return (verification_rank, eligibility_rank[self.eligibility.status],
                verdict_rank[self.verdict],
                self.score if self.score is not None else -1)
