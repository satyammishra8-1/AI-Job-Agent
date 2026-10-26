"""Dashboard services: thin adapters over the SAME functions the CLI uses.

Nothing here reimplements pipeline logic. Search and tailor literally invoke
the CLI command functions with a recording console (same gates, same retries,
same files); the apply preview reuses the runner's building blocks
(``read_form`` → ``build_fill_plan`` → ``apply_drafts``) READ-ONLY — it never
fills the page and never submits. Real submission stays where it was: the
interactive ``job_agent apply --job <id> --submit`` flow with its human gate.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console

from job_agent.apply.tracker import load_applications
from job_agent.store import resolve_apply_url
from job_agent.json_store import read_json

_VERDICT_RANK = {"strong": 3, "possible": 2, "skip": 1, "unscored": 0}
_VERIFICATION_RANK = {"verified": 2, "needs_review": 1, "suspicious": 0}
_ELIGIBILITY_RANK = {"eligible": 2, "review": 1, "not_eligible": 0}


# --- section 1: application pipeline -------------------------------------------------

FOLLOW_UP_TIMEZONE = "Asia/Kolkata"
_FOLLOW_UP_ZONE = timezone(timedelta(hours=5, minutes=30), name=FOLLOW_UP_TIMEZONE)


def _needs_follow_up(record, *, now: datetime | None = None,
                     timezone_name: str = FOLLOW_UP_TIMEZONE) -> bool:
    """True when the record's follow-up date has passed (ISO dates compare
    lexicographically, so no parsing is needed)."""
    # India has a fixed UTC+05:30 offset, so this explicit timezone works on
    # Windows too, where the OS may not provide an IANA timezone database.
    if timezone_name != FOLLOW_UP_TIMEZONE:
        raise ValueError(f"Unsupported follow-up timezone: {timezone_name}")
    current = now or datetime.now(_FOLLOW_UP_ZONE)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("follow-up calculation requires a timezone-aware datetime")
    today = current.astimezone(_FOLLOW_UP_ZONE).date().isoformat()
    return bool(record.follow_up) and record.follow_up <= today


def _next_action(record, *, needs_follow_up: bool) -> str:
    """Return a clear next step based only on the tracked application state."""
    if needs_follow_up:
        return "Follow up now"
    if record.status in {"submitted", "applied", "under_review"} and record.follow_up:
        return f"Follow up by {record.follow_up}"
    return {
        "saved": "Review the role and tailor your resume",
        "ready_to_apply": "Review your resume, then apply",
        "submitted": "Set a follow-up date",
        "applied": "Set a follow-up date",
        "under_review": "Check for an update",
        "assessment": "Prepare for the assessment",
        "interviewing": "Prepare for the interview",
        "offer": "Review the offer",
        "paused": "Resolve the application hold",
        "failed": "Review the failed application attempt",
        "rejected": "No action planned",
        "withdrawn": "No action planned",
    }.get(record.status, "Review application status")


def _dedupe_jobs(records: list) -> list:
    """One row per JOB (records arrive newest-first, so first-seen = latest).

    Collapses rows that logs accumulated before record_attempt deduped on
    write: matched by job_id, else by the (company, title) match-key — the
    same rule the tracker uses for sr-search's re-issued ids.
    """
    from job_agent.apply.tracker import _match_key

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    out = []
    for r in records:
        pair = ((_match_key(r.company), _match_key(r.title))
                if r.company and r.title else None)
        if (r.job_id and r.job_id in seen_ids) or (pair and pair in seen_pairs):
            continue
        if r.job_id:
            seen_ids.add(r.job_id)
        if pair:
            seen_pairs.add(pair)
        out.append(r)
    return out


def applications_view(log_path: str | Path,
                      last_search_path: str | Path | None = None) -> dict:
    """All tracked applications, newest first (one row per job), plus counts."""
    records = _dedupe_jobs(
        sorted(load_applications(log_path), key=lambda r: r.date, reverse=True))
    search_jobs: dict[str, dict] = {}
    if last_search_path is not None:
        data = read_json(last_search_path, {})
        if isinstance(data, dict) and isinstance(data.get("jobs"), dict):
            search_jobs = data["jobs"]
    counts = {"total": len(records)}
    for status in ("submitted", "paused", "failed"):
        counts[status] = sum(1 for r in records if r.status == status)
    counts["needs_follow_up"] = sum(1 for r in records if _needs_follow_up(r))
    return {
        "records": [
            {
                **r.model_dump(),
                "needs_follow_up": _needs_follow_up(r),
                "next_action": _next_action(r, needs_follow_up=_needs_follow_up(r)),
                "job_url": (search_jobs.get(r.job_id) or {}).get("url"),
                "match_score": (search_jobs.get(r.job_id) or {}).get("score"),
            }
            for r in records
        ],
        "counts": counts,
    }


# --- section 2: search + browse -------------------------------------------------------

def jobs_view(last_search_path: str | Path,
              tracker_path: str | Path | None = None, *,
              facts_path: str | Path | None = None,
              out_dir: str | Path | None = None) -> dict:
    """The latest saved search, ranked like the CLI table, with each job's
    tracked state (status / notes / follow-up) joined in by job id."""
    path = Path(last_search_path)
    if not path.exists():
        return {"generated_at": None, "meta": None, "jobs": []}
    data = read_json(path, None)
    if data is None:
        return {"generated_at": None, "meta": None, "jobs": []}
    if not isinstance(data, dict) or not isinstance(data.get("jobs", {}), dict):
        raise ValueError(f"Invalid search data in {path}: expected an object with jobs")
    jobs = list(data.get("jobs", {}).values())
    facts = None
    if facts_path is not None and out_dir is not None:
        from job_agent.tailor.career_facts import load_career_facts

        try:
            facts = load_career_facts(facts_path)
        except (FileNotFoundError, ValueError):
            pass
    tailoring_records = read_json(Path(out_dir) / "tailoring_status.json", {}) \
        if out_dir is not None else {}
    if not isinstance(tailoring_records, dict):
        tailoring_records = {}
    for job in jobs:
        job.setdefault("eligibility", {"status": "review", "reasons": [
            "Eligibility was not assessed for this saved result."]})
        job.setdefault("supported_matches", [])
        job.setdefault("supported_gaps", [])
        job.setdefault("decision", "review")
        if not isinstance(job.get("verification"), dict):
            job["verification"] = {
                "status": "needs_review",
                "signals": [],
                "evidence": {"canonical_url": None, "source_url": job.get("url"),
                             "employer_domain": None, "application_domain": None},
                "reason": "Verification was not recorded for this saved search; review the posting and employer site.",
            }
        if facts is not None and out_dir is not None:
            from job_agent.ats import ats_readiness
            from job_agent.models import Job

            job["ats_readiness"] = ats_readiness(Job.model_validate(job), facts)
            saved_status = tailoring_records.get(str(job.get("id")), {})
            resume_path = _tailored_resume_path(job, facts=facts, out_dir=out_dir)
            valid_record = (isinstance(saved_status, dict)
                            and saved_status.get("job_id") == str(job.get("id")))
            job["resume_status"] = (
                "READY" if valid_record and saved_status.get("status") == "READY"
                and resume_path is not None else
                "NEEDS_REVIEW" if valid_record and saved_status.get("status") == "NEEDS_REVIEW"
                else "NOT_PREPARED")
            job["tailored_resume_ready"] = job["resume_status"] == "READY"
            job["tailoring_reason"] = saved_status.get("reason") if valid_record else None
            job["tailoring_summary"] = saved_status.get("summary", {}) if valid_record else {}
            if valid_record and isinstance(saved_status.get("ats"), dict):
                job["tailoring_ats"] = saved_status["ats"]
        else:
            job["ats_readiness"] = None
            job["resume_status"] = "UNAVAILABLE"
            job["tailored_resume_ready"] = None
    jobs.sort(key=lambda j: (_VERIFICATION_RANK.get(
                                 (j.get("verification") or {}).get("status"), 1),
                             _ELIGIBILITY_RANK.get(
                                 (j.get("eligibility") or {}).get("status"), 1),
                             _VERDICT_RANK.get(j.get("verdict"), 0),
                             j.get("score") if j.get("score") is not None else -1),
              reverse=True)
    tracked: dict[str, dict] = {}
    markers = {"ids": set(), "pairs": set()}
    if tracker_path is not None:
        from job_agent.apply.tracker import applied_markers

        for r in load_applications(tracker_path):     # later records win
            if r.job_id:
                needs_follow_up = _needs_follow_up(r)
                tracked[r.job_id] = {**r.model_dump(),
                                     "needs_follow_up": needs_follow_up,
                                     "next_action": _next_action(
                                         r, needs_follow_up=needs_follow_up)}
        markers = applied_markers(tracker_path)
    from job_agent.apply.tracker import is_already_applied

    for j in jobs:
        j["tracked"] = tracked.get(str(j.get("id")))
        j["already_applied"] = is_already_applied(
            markers, job_id=str(j.get("id")), company=j.get("company", ""),
            title=j.get("title", ""))
    return {"generated_at": data.get("generated_at"),
            "meta": data.get("meta"), "jobs": jobs}


def _tailored_resume_path(record: dict, *, facts, out_dir: str | Path) -> Path | None:
    """Resolve a resume for the given record using the safe, shared filename rule."""
    from job_agent.cli import _job_resume_filename

    root = Path(out_dir).resolve()
    filename = _job_resume_filename((facts.name or "candidate").split()[0],
                                    record.get("title", ""), record.get("company", ""),
                                    str(record.get("id", "")))
    pdf = (root / f"{filename}.pdf").resolve()
    if pdf.parent != root or not pdf.exists():
        return None
    return pdf


def _run_cli(handler, ns: Namespace) -> dict:
    """Run one CLI command function with a recording console (same code path)."""
    console = Console(record=True, width=120)
    try:
        code = handler(console, ns)
    except Exception as exc:  # surfaced to the UI, never a bare 500
        return {"ok": False, "exit_code": 1,
                "output": console.export_text() + f"\nerror: {exc}"}
    return {"ok": code == 0, "exit_code": code, "output": console.export_text()}


def run_search_cli(profile_path: str | Path, days: int = 30) -> dict:
    """Run the real search+score pipeline via the CLI's own command function."""
    from job_agent.cli import cmd_search

    # Must define every attribute cmd_search reads (the dashboard's own
    # "Hide applied" toggle filters client-side, so the printed table hides too).
    ns = Namespace(demo=False, profile=str(profile_path), days=days,
                   max_age_hours=None, limit=None, method="structured",
                   include_applied=False)
    return _run_cli(cmd_search, ns)


def open_in_chrome(url: str, *, _run=None, _fallback=None) -> str:
    """Open ``url`` in the user's real Chrome (their profile, their extensions),
    falling back to the default browser. Returns which opener was used, so the
    UI can say where the page went. ``_run``/``_fallback`` are injectable for
    tests — nothing here is ever headless or automation-controlled."""
    if _run is None:
        import subprocess
        _run = lambda cmd, **kw: subprocess.run(  # noqa: E731 - tiny adapter
            cmd, check=True, capture_output=True, timeout=10, **kw)
    if _fallback is None:
        import webbrowser
        _fallback = webbrowser.open
    try:
        _run(["open", "-a", "Google Chrome", url])
        return "chrome"
    except Exception:
        _fallback(url)
        return "default-browser"


# --- section 3: tailor + apply preview -------------------------------------------------

def run_tailor_cli(job_id: str, *, data_dir: Path, facts_path: Path,
                   out_dir: Path) -> dict:
    """Tailor one job via the CLI's own command function; report the PDF path."""
    from job_agent.cli import _job_resume_filename, cmd_tailor
    from job_agent.store import load_job_record
    from job_agent.tailor.career_facts import load_career_facts

    ns = Namespace(demo=False, job=job_id, facts=str(facts_path), jd=None,
                   out_dir=str(out_dir))
    result = _run_cli(cmd_tailor, ns)
    if result["ok"]:
        record = load_job_record(data_dir / "last_search.json", job_id) or {}
        try:
            facts = load_career_facts(facts_path)
            filename = _job_resume_filename((facts.name or "candidate").split()[0],
                                            record.get("title", ""),
                                            record.get("company", ""), job_id)
            pdf = Path(out_dir) / f"{filename}.pdf"
            if pdf.exists():
                result["pdf"] = str(pdf)
        except (FileNotFoundError, ValueError):
            pass
    return result


def resume_pdf_path(record: dict, *, facts_path: Path, out_dir: Path) -> Path | None:
    """The tailored PDF for this job, or None. NEVER escapes ``out_dir``:
    the filename is derived (slugged) from the stored record — no user path —
    and the resolved location is verified to sit inside the output directory."""
    from job_agent.json_store import read_json
    from job_agent.tailor.career_facts import load_career_facts

    try:
        facts = load_career_facts(facts_path)
    except (FileNotFoundError, ValueError):
        return None
    statuses = read_json(Path(out_dir) / "tailoring_status.json", {})
    saved = statuses.get(str(record.get("id")), {}) if isinstance(statuses, dict) else {}
    if not isinstance(saved, dict) or saved.get("status") != "READY":
        return None
    return _tailored_resume_path(record, facts=facts, out_dir=out_dir)


def serialize_plan(plan) -> dict:
    """A FillPlan as JSON for the UI, tagged exactly like the CLI review."""
    from job_agent.apply.review import _unfilled_tag, source_tag

    def name_of(field) -> str:
        return field.label or field.name or field.selector

    return {
        "planned": [
            {"label": name_of(p.field), "selector": p.field.selector,
             "value": p.value, "source": p.source, "tag": source_tag(p.source),
             "required": p.field.required}
            for p in plan.planned
        ],
        "unfilled": [
            {"label": name_of(u.field), "selector": u.field.selector,
             "reason": u.reason, "tag": _unfilled_tag(u.reason),
             "required": u.field.required}
            for u in plan.unfilled
        ],
        "missing_required": len(plan.missing_required()),
    }


def build_apply_preview(record: dict, bank, contact, *, resume_path: Path | None,
                        drafter, browser_factory=None) -> dict:
    """Read the live form and return the tagged fill plan — WITHOUT filling it.

    Same building blocks as the runner (read_form → build_fill_plan →
    apply_drafts) and the same review tags the CLI shows. The page is only
    navigated and scanned; no value is entered and nothing is submitted — the
    returned ``submit_command`` is the human-gated CLI flow.
    """
    from job_agent.apply.browser import open_browser
    from job_agent.apply.filler import build_fill_plan
    from job_agent.apply.form_reader import read_form
    from job_agent.apply.screening import apply_drafts

    factory = browser_factory or open_browser
    with factory(headless=True) as page:
        page.goto(resolve_apply_url(record), wait_until="load")
        fields = read_form(page)

    plan = build_fill_plan(fields, bank, contact, resume_path)
    if drafter is not None:
        plan = apply_drafts(plan, drafter)

    return {
        "job": {"id": record.get("id"), "title": record.get("title"),
                "company": record.get("company")},
        **serialize_plan(plan),
        "submit_command": f"python -m job_agent apply --job {record.get('id')} --submit",
    }


def _assemble_apply_inputs(job_id: str, *, data_dir: Path, facts_path: Path,
                           out_dir: Path) -> dict:
    """The real inputs every apply flavour needs: record, bank, contact,
    tailored resume, and (with an API key) the grounded screening drafter."""
    from job_agent.apply.answer_bank import load_answer_bank, resolve_contact
    from job_agent.cli import _find_tailored_resume
    from job_agent.config import load_settings
    from job_agent.store import load_job_record
    from job_agent.tailor.career_facts import load_career_facts

    settings = load_settings()
    record = load_job_record(data_dir / "last_search.json", job_id)
    if record is None:
        raise KeyError(job_id)
    facts = load_career_facts(facts_path)
    bank = load_answer_bank(data_dir / "answer_bank.yaml")
    contact = resolve_contact(facts, bank)
    resume = _find_tailored_resume(None, Path(out_dir), facts, record)

    drafter = None
    if settings.gemini_api_key:
        from job_agent.apply.screening import make_drafter, make_llm_generate
        drafter = make_drafter(
            make_llm_generate(settings), facts, bank,
            jd=record.get("description") or "", company=record.get("company", ""),
            cache_path=data_dir / "answers_cache.json")
    return {"record": record, "bank": bank, "contact": contact,
            "resume_path": resume, "drafter": drafter}


def run_apply_preview(job_id: str, *, data_dir: Path, facts_path: Path,
                      out_dir: Path) -> dict:
    """Assemble the real inputs (record, bank, facts, resume, drafter) and preview."""
    inputs = _assemble_apply_inputs(job_id, data_dir=data_dir,
                                    facts_path=facts_path, out_dir=out_dir)
    record = inputs.pop("record")
    return build_apply_preview(record, inputs.pop("bank"), inputs.pop("contact"),
                               **inputs)


def start_apply_session(job_id: str, *, data_dir: Path, facts_path: Path,
                        out_dir: Path, verification_reviewed: bool = False):
    """Build a live, human-gated apply session (visible browser) for one job."""
    from job_agent.dashboard.apply_session import ApplySession

    inputs = _assemble_apply_inputs(job_id, data_dir=data_dir,
                                    facts_path=facts_path, out_dir=out_dir)
    return ApplySession(tracker_path=data_dir / "applications.json",
                        out_dir=data_dir / "apply",
                        verification_reviewed=verification_reviewed, **inputs)
