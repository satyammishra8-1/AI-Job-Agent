"""Explicit, read-only discovery audit reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from job_agent.seen_cache import SeenCache


class ReadOnlySeenCache(SeenCache):
    """Load the user's seen-job history for accurate freshness, never save it."""

    def save(self) -> None:
        return None


@dataclass
class SourceAudit:
    requests: int = 0
    pages_fetched: int = 0
    jobs_returned: int = 0
    errors: list[str] = field(default_factory=list)
    rate_limits: int = 0
    response_time_ms: float = 0.0


class AuditCollector:
    """In-memory counters used only by ``search --audit``."""

    def __init__(self) -> None:
        self.sources: dict[str, SourceAudit] = {}
        self.stages: dict[str, int] = {}
        self.rejections: Counter[str] = Counter()
        self.rejection_details: Counter[str] = Counter()

    def _source(self, source: str) -> SourceAudit:
        return self.sources.setdefault(source, SourceAudit())

    def record_request(self, source: str, phase: str, status: int | None,
                       elapsed: float, error: str | None) -> None:
        item = self._source(source)
        item.requests += 1
        item.response_time_ms += elapsed * 1000
        if phase == "fetch" and status == 200:
            # Each successful GET made by fetch() is one returned API page.
            item.pages_fetched += 1
        if status == 429:
            item.rate_limits += 1
        if error:
            item.errors.append(error)

    def source_failure(self, source: str, message: str) -> None:
        item = self._source(source)
        if not item.errors or item.errors[-1] != message:
            item.errors.append(message)

    def stage(self, name: str, count: int) -> None:
        self.stages[name] = count

    def reject(self, reason: str, count: int) -> None:
        if count > 0:
            self.rejections[reason] += count

    def reject_detail(self, reason: str, count: int = 1) -> None:
        if count > 0:
            self.rejection_details[reason] += count

    def set_jobs_returned(self, per_source: dict[str, int]) -> None:
        for source, count in per_source.items():
            self._source(source).jobs_returned = count

    def quality_flags(self, low_result_threshold: int = 5) -> dict[str, list[str]]:
        return {
            "low_results": [name for name, source in self.sources.items()
                            if source.jobs_returned < low_result_threshold],
            "failing": [name for name, source in self.sources.items() if source.errors],
        }


def is_internship_or_entry_level(title: str) -> bool:
    """Audit classifier only; it never changes normal discovery results."""
    return bool(re.search(
        r"\b(intern(?:ship)?|graduate|new\s*grad|entry[ -]?level|fresher|trainee|junior)\b",
        title or "", re.IGNORECASE,
    ))


def build_report(outcome, collector: AuditCollector, scored_jobs) -> dict:
    """Build the final audit stages without mutating search or user stores."""
    collector.set_jobs_returned(outcome.per_source)
    candidates = [job for job in outcome.jobs if is_internship_or_entry_level(job.title)]
    collector.stage("after_internship_entry_filter", len(candidates))
    collector.reject("internship_entry_filter", len(outcome.jobs) - len(candidates))
    collector.reject_detail(
        "title not classified as internship, graduate, entry-level, fresher, trainee, or junior",
        len(outcome.jobs) - len(candidates),
    )

    verified = [job for job in candidates
                if job.verification and job.verification.status == "verified"]
    collector.stage("after_verification", len(verified))
    collector.reject("verification", len(candidates) - len(verified))
    for job in candidates:
        if not job.verification or job.verification.status != "verified":
            collector.reject_detail(
                job.verification.reason if job.verification else "verification result missing")

    scored_by_id = {str(item.job.id): item for item in scored_jobs}
    eligible = [scored_by_id[str(job.id)] for job in verified
                if str(job.id) in scored_by_id
                and scored_by_id[str(job.id)].eligibility.status == "eligible"]
    collector.stage("after_eligibility", len(eligible))
    collector.reject("eligibility", len(verified) - len(eligible))
    for item in (scored_by_id.get(str(job.id)) for job in verified):
        if item and item.eligibility.status != "eligible":
            details = item.eligibility.reasons or (item.eligibility.status,)
            for reason in details:
                collector.reject_detail(reason)

    matched = [item for item in eligible
               if item.decision == "recommended" and item.verdict in {"strong", "possible"}]
    collector.stage("after_profile_matching", len(matched))
    collector.reject("profile_matching", len(eligible) - len(matched))
    for item in eligible:
        if item not in matched:
            collector.reject_detail(
                f"profile matching decision={item.decision}, verdict={item.verdict}")

    return {
        "read_only": True,
        "stages": {**collector.stages,
                   "after_experience": outcome.counts.after_experience},
        "sources": {
            name: {
                "requests": source.requests,
                "pages_fetched": source.pages_fetched,
                "jobs_returned": source.jobs_returned,
                "errors": source.errors,
                "rate_limits": source.rate_limits,
                "response_time_ms": round(source.response_time_ms, 1),
            }
            for name, source in collector.sources.items()
        },
        "rejections": dict(collector.rejections),
        "rejection_details": dict(collector.rejection_details),
        "source_health": collector.quality_flags(),
        "filter_notes": [
            "Internship/entry-level classification is title-only and used for audit reporting; it does not change normal search behavior.",
            "Verification, eligibility, and matching use existing application rules without relaxation.",
            "Low-result threshold is fewer than five jobs returned for a configured source in this run.",
        ],
    }
