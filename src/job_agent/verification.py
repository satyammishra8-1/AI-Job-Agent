"""Deterministic, conservative checks for job-posting provenance and risk.

The job description is untrusted data. This module never sends it to an LLM,
follows links, or treats posting text as instructions. A per-company ATS board
response is evidence that the posting exists on that configured board; a
cross-company feed is not canonical employer evidence.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from job_agent.models import (
    Job,
    VerificationEvidence,
    VerificationResult,
    VerificationSignal,
)
from job_agent.store import validate_application_url

_EMPLOYER_ATS_SOURCES = frozenset({
    "greenhouse", "lever", "ashby", "smartrecruiters",
})
_ATS_DOMAINS = (
    "greenhouse.io", "lever.co", "ashbyhq.com", "smartrecruiters.com",
)
_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rebrand.ly",
    "shorturl.at", "rb.gy", "tiny.cc",
})
_UPFRONT_PAYMENT = re.compile(
    r"\b(?:pay\s+(?:to\s+apply|before\s+(?:an?\s+)?interview)|"
    r"(?:application|registration|processing|security\s+deposit|training|"
    r"certification)\s+(?:fee|charge|deposit)|"
    r"(?:fee|deposit)\s+(?:is\s+)?required\s+(?:to\s+)?(?:apply|interview)|"
    r"pay\s+for\s+(?:mandatory\s+)?(?:training|certification))\b",
    re.IGNORECASE,
)
_SENSITIVE_EARLY = re.compile(
    r"\b(?:aadhaar|aadhar|pan\s+card|bank\s+account|credit\s+card|"
    r"debit\s+card|one[- ]time\s+password|\botp\b)\b",
    re.IGNORECASE,
)
_EARLY_REQUEST = re.compile(
    r"\b(?:send|share|provide|submit|upload|whatsapp)\b.{0,100}"
    r"\b(?:before\s+(?:the\s+)?(?:interview|offer|selection)|"
    r"to\s+(?:apply|schedule\s+(?:an?\s+)?interview))\b",
    re.IGNORECASE,
)
_WHATSAPP_ONLY = re.compile(
    r"\b(?:whatsapp|telegram)\b.{0,60}\b(?:only|exclusively)\b|"
    r"\b(?:only|exclusively)\b.{0,60}\b(?:whatsapp|telegram)\b",
    re.IGNORECASE,
)
_GUARANTEED_PLACEMENT = re.compile(
    r"\b(?:guaranteed\s+(?:job|placement|income)|100\s*%\s*placement)\b",
    re.IGNORECASE,
)
_NO_SCREENING = re.compile(
    r"\b(?:no\s+(?:interview|experience)(?:\s+or\s+(?:interview|experience))?\s+"
    r"(?:is\s+)?required|"
    r"without\s+(?:an?\s+)?interview)\b",
    re.IGNORECASE,
)


def _host(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return None
        return parts.hostname.rstrip(".").lower()
    except ValueError:
        return None


def _is_ats_domain(host: str | None) -> bool:
    return bool(host and any(host == domain or host.endswith("." + domain)
                             for domain in _ATS_DOMAINS))


def _company_matches_host(company: str, host: str | None) -> bool:
    if not host:
        return False
    ignored = {"inc", "llc", "ltd", "limited", "corp", "corporation",
               "company", "technologies", "technology", "solutions", "group"}
    tokens = [re.sub(r"[^a-z0-9]", "", part.lower())
              for part in company.split()]
    tokens = [token for token in tokens if len(token) >= 4 and token not in ignored]
    return any(token in host.replace("-", "") for token in tokens)


def verify_job(job: Job) -> VerificationResult:
    """Assess source evidence and conservative risk indicators without network calls."""
    signals: list[VerificationSignal] = []
    source_host = _host(job.url)
    application_url = job.apply_url or job.url
    application_host = _host(application_url)
    direct_source = job.source in _EMPLOYER_ATS_SOURCES
    canonical_url = job.url if direct_source and source_host else None
    employer_domain = source_host if source_host and not _is_ats_domain(source_host) else None

    if canonical_url:
        signals.append(VerificationSignal(
            code="employer_ats_posting", severity="info",
            message=f"Posting was returned by the employer's configured {job.source} board."))
    else:
        signals.append(VerificationSignal(
            code="canonical_posting_unconfirmed", severity="review",
            message="No canonical employer posting was confirmed by this discovery source."))

    if not source_host:
        signals.append(VerificationSignal(
            code="posting_url_unusable", severity="review",
            message="The posting URL is missing or is not a public HTTP(S) URL."))
    elif (direct_source and not _is_ats_domain(source_host)
          and not _company_matches_host(job.company, source_host)):
        signals.append(VerificationSignal(
            code="employer_domain_mismatch", severity="review",
            message="The posting domain could not be linked to the named employer."))

    if not application_host:
        signals.append(VerificationSignal(
            code="application_url_missing", severity="review",
            message="No usable application destination was provided."))
    else:
        try:
            validate_application_url(application_url)
        except ValueError:
            signals.append(VerificationSignal(
                code="unsafe_application_domain", severity="risk",
                message="The application URL targets an unsafe, malformed, or non-public destination."))

        if application_host in _SHORTENERS:
            signals.append(VerificationSignal(
                code="hidden_application_destination", severity="review",
                message="The application link hides its destination behind a URL shortener."))

        if direct_source:
            same_site = application_host == source_host
            ats_host = _is_ats_domain(application_host)
            if not same_site and not ats_host:
                signals.append(VerificationSignal(
                    code="application_domain_mismatch", severity="review",
                    message="The application destination differs from the posting and is not a recognized ATS domain."))
        elif (not _is_ats_domain(application_host)
              and not _company_matches_host(job.company, application_host)):
            signals.append(VerificationSignal(
                code="employer_domain_mismatch", severity="review",
                message="The application domain could not be linked to the named employer."))

    description = job.description or ""
    if _UPFRONT_PAYMENT.search(description):
        signals.append(VerificationSignal(
            code="upfront_payment_request", severity="risk",
            message="The posting asks the candidate to pay a fee, deposit, or training cost."))
    if _SENSITIVE_EARLY.search(description) and _EARLY_REQUEST.search(description):
        signals.append(VerificationSignal(
            code="sensitive_data_request", severity="risk",
            message="The posting appears to request sensitive personal or financial data before an offer."))
    if _WHATSAPP_ONLY.search(description):
        signals.append(VerificationSignal(
            code="messaging_only_contact", severity="review",
            message="The posting appears to restrict recruitment contact to a messaging service."))
    if _GUARANTEED_PLACEMENT.search(description) and _NO_SCREENING.search(description):
        signals.append(VerificationSignal(
            code="unrealistic_job_promise", severity="risk",
            message="The posting promises guaranteed placement while claiming no screening is required."))

    risk_count = sum(signal.severity == "risk" for signal in signals)
    has_review_signal = any(signal.severity == "review" for signal in signals)
    if risk_count:
        status = "suspicious"
        reason = "; ".join(signal.message for signal in signals
                            if signal.severity == "risk")
    elif direct_source and canonical_url and not has_review_signal:
        status = "verified"
        reason = (f"Posting is present on the employer's configured {job.source} board; "
                  "no high-risk signals were detected.")
    else:
        status = "needs_review"
        reason = next((signal.message for signal in signals
                       if signal.severity == "review"),
                      "The available source evidence is not enough to verify the employer posting.")

    return VerificationResult(
        status=status,
        signals=tuple(signals),
        evidence=VerificationEvidence(
            canonical_url=canonical_url,
            source_url=job.url or None,
            employer_domain=employer_domain,
            application_domain=application_host,
        ),
        reason=reason,
    )
