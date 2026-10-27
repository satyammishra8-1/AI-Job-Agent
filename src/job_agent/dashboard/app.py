"""The local dashboard app: FastAPI over the CLI's own functions.

LOCAL ONLY by design — this serves personal data (application history, career
facts). The CLI command binds it to 127.0.0.1 and there is no auth layer, so
it must never be exposed beyond localhost. No secrets reach the frontend: the
backend reads ``.env`` exactly as the CLI does.
"""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
import yaml

from job_agent.apply.tracker import Status, record_event, upsert_job_state
from job_agent.dashboard import service
from job_agent.store import load_job_record, resolve_apply_url
from job_agent.tailor.career_facts import CareerFacts

_STATIC = Path(__file__).resolve().parent / "static"


class SearchRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=90)


class JobRequest(BaseModel):
    job_id: str = Field(min_length=1)


class ApplyJobRequest(JobRequest):
    verification_reviewed: bool = False


class PreparationEditRequest(JobRequest):
    answers: dict[str, str] = Field(default_factory=dict)
    recruiter_message: str | None = None
    cover_note: str | None = None


class PreparationApprovalRequest(JobRequest):
    reviewed: bool = True


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class SubmitSessionRequest(SessionRequest):
    approval_token: str = Field(min_length=1, max_length=100)


class EditRequest(SessionRequest):
    selector: str = Field(min_length=1)
    value: str
    mark_done: bool = False


class TrackRequest(BaseModel):
    """User-managed state for one job: status pipeline, notes, follow-up date.

    Keyed by ``job_id`` (creates the record on first use — a manually-submitted
    application needs no prior apply run), or by ``attempt_id`` for old records
    that carry no job id.
    """

    job_id: str = ""
    attempt_id: str = ""
    company: str = ""
    title: str = ""
    source: str = ""
    status: Status | None = None
    notes: str | None = None
    follow_up: str | None = Field(default=None, pattern=r"^(\d{4}-\d{2}-\d{2})?$")
    follow_up_note: str | None = None
    follow_up_completed: bool | None = None

    @model_validator(mode="after")
    def _keyed(self) -> "TrackRequest":
        if not self.job_id and not self.attempt_id:
            raise ValueError("job_id or attempt_id is required")
        return self


class TrackEventRequest(BaseModel):
    attempt_id: str = Field(min_length=1)
    event: str = Field(pattern=r"^(follow_up_completed|interview|assessment|offer|rejection|note)$")
    date: str | None = Field(default=None, pattern=r"^(\d{4}-\d{2}-\d{2})?$")
    kind: str = ""
    notes: str = ""
    outcome: str = ""
    reason: str = ""


def create_app(*, data_dir: Path = Path("data"),
               profile_path: Path = Path("search_profile.yaml"),
               facts_path: Path | None = None,
               out_dir: Path | None = None,
               searcher=None, tailorer=None, previewer=None,
               apply_session_factory=None, url_opener=None) -> FastAPI:
    """Build the app. ``searcher``/``tailorer``/``previewer``/
    ``apply_session_factory`` are injectable for tests; the defaults call the
    real service functions (i.e. the CLI paths)."""
    data_dir = Path(data_dir)
    facts_path = Path(facts_path or data_dir / "career_facts.yaml")
    out_dir = Path(out_dir or data_dir / "output")
    searcher = searcher or (lambda days: service.run_search_cli(profile_path, days,
                                                                  facts_path=facts_path))
    tailorer = tailorer or (lambda job_id: service.run_tailor_cli(
        job_id, data_dir=data_dir, facts_path=facts_path, out_dir=out_dir))
    previewer = previewer or (lambda job_id: service.run_apply_preview(
        job_id, data_dir=data_dir, facts_path=facts_path, out_dir=out_dir))
    if apply_session_factory is None:
        apply_session_factory = (
            lambda job_id, *, verification_reviewed=False: service.start_apply_session(
                job_id, data_dir=data_dir, facts_path=facts_path, out_dir=out_dir,
                verification_reviewed=verification_reviewed))
    else:
        injected_factory = apply_session_factory
        apply_session_factory = (
            lambda job_id, *, verification_reviewed=False: injected_factory(job_id))
    url_opener = url_opener or service.open_in_chrome

    app = FastAPI(title="job-agent dashboard", docs_url=None, redoc_url=None)
    # The Chrome extension calls this app cross-origin; nothing else may.
    from fastapi.middleware.cors import CORSMiddleware

    from job_agent.dashboard.extension_api import (
        cors_config, create_extension_router, is_trusted_extension_origin,
    )

    @app.middleware("http")
    async def reject_cross_origin_mutations(request, call_next):
        """CORS is not CSRF protection; validate browser Origin on mutations."""
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            # Non-browser local clients do not send Origin. Browser requests
            # must originate in this local UI or the configured extension.
            if origin:
                from urllib.parse import urlsplit

                parts = urlsplit(origin)
                local_ui = (parts.scheme == "http" and parts.hostname in
                            {"127.0.0.1", "localhost", "::1"})
                if not local_ui and not is_trusted_extension_origin(origin):
                    return JSONResponse({"detail": "untrusted request origin"}, status_code=403)
        return await call_next(request)

    app.add_middleware(CORSMiddleware, **cors_config())
    # Shared with the extension router: the real-browser apply channel.
    pending_task: dict = {}            # {"task": {...}} set by /api/apply/open
    fill_reports: dict = {}            # job_id -> last extension fill report
    app.include_router(create_extension_router(data_dir=data_dir, facts_path=facts_path,
                                               out_dir=out_dir,
                                               pending_task=pending_task,
                                               fill_reports=fill_reports))
    sessions: dict[str, object] = {}   # live apply sessions, one browser each
    session_jobs: dict[str, str] = {}
    session_preparations: dict[str, str] = {}

    def _require_job(job_id: str) -> dict:
        record = load_job_record(data_dir / "last_search.json", job_id)
        if record is None:
            raise HTTPException(404, f"job id {job_id!r} not in the last search")
        return record

    def _require_suspicious_review(record: dict, acknowledged: bool) -> None:
        verification = record.get("verification") or {}
        if (isinstance(verification, dict)
                and verification.get("status") == "suspicious" and not acknowledged):
            reason = verification.get("reason") or "Risk signals were detected."
            raise HTTPException(
                409,
                f"Suspicious job requires explicit review before opening an application: {reason}",
            )

    preparation_path = data_dir / "application_preparations.json"

    def _invalidate_tracked_approval(job_id: str) -> None:
        from job_agent.apply.tracker import load_applications, update_application

        records = load_applications(data_dir / "applications.json")
        tracked = next((r for r in reversed(records)
                       if r.job_id == job_id and r.status == "ready_to_apply"
                       and r.approval_state == "approved"), None)
        if tracked is None:
            return
        paused = upsert_job_state(data_dir / "applications.json", job_id=job_id,
                                  status="paused")
        update_application(data_dir / "applications.json", paused.attempt_id,
            approval_state="invalidated", approval_identity="",
            reason="Prepared content or job context changed; review required")

    def _require_prepared_approval(job_id: str) -> str:
        from job_agent.apply.preparation import get_preparation, is_approved
        from job_agent.json_store import read_json, write_json_atomic
        prep = get_preparation(preparation_path, job_id)
        record = _require_job(job_id)
        resume = service.resume_pdf_path(record, facts_path=facts_path, out_dir=out_dir)
        verification = (record.get("verification") or {}).get("status")
        eligibility = (record.get("eligibility") or {}).get("status")
        if (verification != "verified" or eligibility != "eligible"
                or not is_approved(prep, job=record, resume_path=resume)):
            if prep.get("status") == "READY_TO_APPLY":
                prep["status"] = "NEEDS_REVIEW"
                prep["reviewed"] = False
                prep["approved_fingerprint"] = None
                data = read_json(preparation_path, {})
                if isinstance(data, dict):
                    data[job_id] = prep
                    write_json_atomic(preparation_path, data)
            _invalidate_tracked_approval(job_id)
            raise HTTPException(409, "Application preparation must be reviewed and approved in the dashboard first")
        return prep["approved_fingerprint"]

    @app.get("/api/applications")
    def applications() -> dict:
        return service.applications_view(data_dir / "applications.json",
                                         data_dir / "last_search.json")

    @app.get("/api/dashboard/summary")
    def dashboard_summary() -> dict:
        return service.dashboard_summary(data_dir / "applications.json")

    @app.get("/api/jobs")
    def jobs() -> dict:
        return service.jobs_view(data_dir / "last_search.json",
                                 data_dir / "applications.json",
                                 facts_path=facts_path, out_dir=out_dir)

    @app.get("/api/profile")
    def profile() -> dict:
        from job_agent.tailor.career_facts import load_career_facts

        if not facts_path.exists():
            facts = CareerFacts()
            from job_agent.profile_intelligence import analyze_profile
            return {"configured": False, "profile": facts.model_dump(mode="json"),
                    "analysis": analyze_profile(facts)}
        try:
            facts = load_career_facts(facts_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        from job_agent.profile_intelligence import analyze_profile
        return {"configured": facts.profile_configured(),
                "profile": facts.model_dump(mode="json"),
                "analysis": analyze_profile(facts)}

    @app.get("/api/profile/analyze")
    def profile_analysis() -> dict:
        from job_agent.profile_intelligence import analyze_profile
        from job_agent.tailor.career_facts import load_career_facts
        try:
            facts = load_career_facts(facts_path)
        except FileNotFoundError:
            facts = CareerFacts()
        return analyze_profile(facts)

    @app.put("/api/profile")
    def save_profile(req: CareerFacts) -> dict:
        from job_agent.json_store import read_json, write_json_atomic
        from job_agent.profile_intelligence import analyze_profile
        from job_agent.tailor.career_facts import load_career_facts
        requested = CareerFacts.model_validate(req.model_dump())
        existing = load_career_facts(facts_path) if facts_path.exists() else CareerFacts()
        old_content = existing.model_dump(mode="json", exclude={"profile_id", "profile_version"})
        new_content = requested.model_dump(mode="json", exclude={"profile_id", "profile_version"})
        changed = (not facts_path.exists()) or old_content != new_content
        facts = requested.model_copy(update={
            "profile_id": existing.profile_id or requested.profile_id or uuid4().hex,
            "profile_version": existing.profile_version + 1 if changed else
                              max(existing.profile_version, requested.profile_version),
        })
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=facts_path.parent,
                prefix=f".{facts_path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = stream.name
                yaml.safe_dump(facts.model_dump(mode="json", exclude_defaults=True),
                               stream, sort_keys=False, allow_unicode=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, facts_path)
            temporary = None
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        if changed and preparation_path.exists():
            preparations = read_json(preparation_path, {})
            if isinstance(preparations, dict):
                for prep in preparations.values():
                    if isinstance(prep, dict) and prep.get("status") in {"READY_TO_APPLY", "ANSWERS_READY"}:
                        prep.update(status="NEEDS_REVIEW", reviewed=False,
                                    approved_fingerprint=None, profile_change_review_required=True)
                write_json_atomic(preparation_path, preparations)
            from job_agent.apply.tracker import load_applications
            for record in load_applications(data_dir / "applications.json"):
                if record.status == "ready_to_apply" and record.approval_state == "approved":
                    _invalidate_tracked_approval(record.job_id)
        if changed:
            from job_agent.json_store import read_json, write_json_atomic
            tailoring_path = out_dir / "tailoring_status.json"
            tailoring = read_json(tailoring_path, {}) if tailoring_path.exists() else None
            if isinstance(tailoring, dict):
                for item in tailoring.values():
                    if isinstance(item, dict) and item.get("status") == "READY":
                        item.update(status="NEEDS_REVIEW",
                                    reason="Developer profile changed; tailor and validate this resume again.")
                write_json_atomic(tailoring_path, tailoring)
        return {"configured": facts.profile_configured(),
                "profile": facts.model_dump(mode="json"),
                "analysis": analyze_profile(facts), "changed": changed}

    @app.post("/api/search")
    def run_search(req: SearchRequest) -> dict:
        result = searcher(req.days)
        # whatever the run printed, the table shows what's now on disk
        return {**result, **service.jobs_view(data_dir / "last_search.json",
                                              data_dir / "applications.json",
                                              facts_path=facts_path, out_dir=out_dir)}

    @app.post("/api/track")
    def track(req: TrackRequest) -> dict:
        try:
            record = upsert_job_state(
                data_dir / "applications.json", job_id=req.job_id,
                attempt_id=req.attempt_id, company=req.company,
                title=req.title, source=req.source, status=req.status,
                notes=req.notes, follow_up=req.follow_up,
                follow_up_note=req.follow_up_note,
                follow_up_completed=req.follow_up_completed)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {**record.model_dump(),
                "needs_follow_up": service._needs_follow_up(record)}

    @app.post("/api/track/event")
    def track_event(req: TrackEventRequest) -> dict:
        from datetime import datetime, timezone
        from job_agent.apply.tracker import load_applications

        try:
            current = next(r for r in load_applications(data_dir / "applications.json")
                           if r.attempt_id == req.attempt_id)
        except StopIteration as exc:
            raise HTTPException(404, "Application not found") from exc
        stamp = req.date or datetime.now(timezone.utc).date().isoformat()
        mapping = {
            "follow_up_completed": ("Follow-up completed", {"follow_up_completed": True}),
            "interview": ("Interview", {"interview_date": stamp, "interview_type": req.kind,
                "interview_notes": req.notes, "interview_outcome": req.outcome,
                "status": "interviewing"}),
            "assessment": ("Assessment", {"assessment_date": stamp, "assessment_type": req.kind,
                "assessment_notes": req.notes, "assessment_outcome": req.outcome,
                "status": "assessment"}),
            "offer": ("Offer", {"offer_date": stamp, "offer_notes": req.notes,
                "offer_outcome": req.outcome, "status": "offer"}),
            "rejection": ("Rejection", {"rejection_date": stamp,
                "rejection_reason": req.reason, "rejection_notes": req.notes,
                "status": "rejected"}),
            "note": ("Note added", {"notes": req.notes}),
        }
        label, fields = mapping[req.event]
        if "status" in fields:
            from job_agent.apply.tracker import transition_allowed
            if not transition_allowed(current.status, fields["status"]):
                raise HTTPException(409, f"Invalid status transition: {current.status} -> {fields['status']}")
        updated = record_event(data_dir / "applications.json", req.attempt_id, label, **fields)
        return {**updated.model_dump(), "needs_follow_up": service._needs_follow_up(updated)}

    @app.get("/api/resume/{job_id}")
    def resume(job_id: str) -> FileResponse:
        record = _require_job(job_id)
        pdf = service.resume_pdf_path(record, facts_path=facts_path, out_dir=out_dir)
        if pdf is None:
            raise HTTPException(404, "no tailored resume for this job — run Tailor first")
        return FileResponse(pdf, media_type="application/pdf")

    @app.post("/api/tailor")
    def run_tailor(req: JobRequest) -> dict:
        _require_job(req.job_id)
        return tailorer(req.job_id)

    @app.get("/api/preparation/{job_id}")
    def application_preparation(job_id: str) -> dict:
        from job_agent.apply.preparation import get_preparation
        _require_job(job_id)
        return get_preparation(preparation_path, job_id)

    @app.post("/api/preparation/generate")
    def prepare_application(req: JobRequest) -> dict:
        from job_agent.apply.answer_bank import load_answer_bank
        from job_agent.apply.preparation import generate_preparation, get_preparation
        from job_agent.config import load_settings
        from job_agent.tailor.career_facts import load_career_facts
        record = _require_job(req.job_id)
        view = service.jobs_view(data_dir / "last_search.json", facts_path=facts_path,
                                 out_dir=out_dir)
        job = next((j for j in view["jobs"] if str(j.get("id")) == req.job_id), record)
        try:
            prior = get_preparation(preparation_path, req.job_id)
            facts = load_career_facts(facts_path)
            bank = load_answer_bank(data_dir / "answer_bank.yaml")
            resume = service.resume_pdf_path(record, facts_path=facts_path, out_dir=out_dir)
            result = generate_preparation(path=preparation_path, job=job, facts=facts,
                bank=bank, resume_status=job.get("resume_status", "NOT_PREPARED"),
                settings=load_settings(), jd=record.get("description") or "",
                resume_path=resume)
            if prior.get("status") == "READY_TO_APPLY":
                _invalidate_tracked_approval(req.job_id)
            return result
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.put("/api/preparation/edit")
    def edit_preparation(req: PreparationEditRequest) -> dict:
        from job_agent.apply.answer_bank import load_answer_bank
        from job_agent.apply.preparation import update_preparation
        from job_agent.tailor.career_facts import load_career_facts
        updates = {"answers": req.answers}
        if req.recruiter_message is not None: updates["recruiter_message"] = req.recruiter_message
        if req.cover_note is not None: updates["cover_note"] = req.cover_note
        try:
            record = _require_job(req.job_id)
            facts = load_career_facts(facts_path)
            bank = load_answer_bank(data_dir / "answer_bank.yaml")
            result = update_preparation(preparation_path, req.job_id, updates,
                                        facts=facts, company=record.get("company", ""),
                                        bank=bank)
            _invalidate_tracked_approval(req.job_id)
            return result
        except KeyError as exc:
            raise HTTPException(404, "Application preparation not found") from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/preparation/approve")
    def approve_application(req: PreparationApprovalRequest) -> dict:
        import hashlib
        from job_agent.apply.preparation import (approve_preparation,
            get_preparation, _context_fingerprint)
        from job_agent.apply.tracker import update_application
        from job_agent.store import resolve_apply_url
        try:
            job = _require_job(req.job_id)
            if ((job.get("verification") or {}).get("status") != "verified"
                    or (job.get("eligibility") or {}).get("status") != "eligible"):
                raise HTTPException(409, "Only verified, eligible opportunities can be approved")
            prep = get_preparation(preparation_path, req.job_id)
            resume = service.resume_pdf_path(job, facts_path=facts_path, out_dir=out_dir)
            if prep.get("context_fingerprint") != _context_fingerprint(job, resume):
                from job_agent.json_store import read_json, write_json_atomic
                prep["status"] = "NEEDS_REVIEW"
                prep["reviewed"] = False
                prep["approved_fingerprint"] = None
                preparations = read_json(preparation_path, {})
                if isinstance(preparations, dict):
                    preparations[req.job_id] = prep
                    write_json_atomic(preparation_path, preparations)
                _invalidate_tracked_approval(req.job_id)
                raise HTTPException(409, "Job or resume changed; regenerate and review this application again")
            try:
                application_url = resolve_apply_url(job)
            except ValueError as exc:
                raise HTTPException(409, "This opportunity has an unsafe application URL") from exc
            if not application_url:
                raise HTTPException(409, "This opportunity has no safe application URL")
            approved = approve_preparation(preparation_path, req.job_id)
            tracked = upsert_job_state(
                data_dir / "applications.json", job_id=req.job_id,
                company=job.get("company", "Unknown"), title=job.get("title", ""),
                source=job.get("source", ""), status="ready_to_apply")
            update_application(data_dir / "applications.json", tracked.attempt_id,
                application_url=application_url,
                verification_status=(job.get("verification") or {}).get("status", ""),
                eligibility_status=(job.get("eligibility") or {}).get("status", ""),
                match_score=job.get("score"),
                profile_id=approved.get("profile_id", ""),
                profile_version=approved.get("profile_version", 0),
                resume_identity=hashlib.sha256(resume.read_bytes()).hexdigest() if resume else "",
                resume_version=approved.get("context_fingerprint", ""),
                answers_identity=approved.get("approved_fingerprint", ""),
                answers_version=approved.get("approved_fingerprint", ""),
                submitted_answers={a.get("key", ""): a.get("answer", "")
                                   for a in approved.get("answers", [])},
                recruiter_message=approved.get("recruiter_message", ""),
                cover_note=approved.get("cover_note", ""),
                approval_identity=approved.get("approved_fingerprint", ""),
                approval_state="approved")
            return approved
        except KeyError as exc:
            raise HTTPException(404, "Application preparation not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/apply/preview")
    def apply_preview(req: JobRequest) -> dict:
        _require_job(req.job_id)
        try:
            return previewer(req.job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # --- apply in the user's REAL browser (extension does the filling) -----

    @app.post("/api/apply/open")
    def apply_open(req: ApplyJobRequest) -> dict:
        """Open the job's stored apply URL in the user's own browser and queue
        a fill task for the extension. No Playwright, no filling, no
        submission here — the human drives everything from their browser."""
        record = _require_job(req.job_id)
        _require_suspicious_review(record, req.verification_reviewed)
        try:
            url = resolve_apply_url(record)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not url:
            raise HTTPException(404, "this job record has no apply URL")
        preparation_fingerprint = _require_prepared_approval(req.job_id)
        from datetime import datetime, timezone

        pending_task["task"] = {
            "job_id": req.job_id,
            "company": record.get("company", ""),
            "title": record.get("title", ""),
            "apply_url": url,
            "jd": (record.get("description") or "")[:25_000],
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        opened_via = url_opener(url)
        return {"ok": True, "opened_via": opened_via, "apply_url": url}

    @app.get("/api/apply/fill-report/{job_id}")
    def apply_fill_report(job_id: str) -> dict:
        return {"report": fill_reports.get(job_id)}

    # --- live apply sessions: the CLI's review gate, in the UI -------------

    def _session(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "no such apply session (it may have finished)")
        return session

    @app.post("/api/apply/start")
    def apply_start(req: ApplyJobRequest) -> dict:
        from uuid import uuid4
        from job_agent.dashboard.apply_session import SubmitBlocked

        record = _require_job(req.job_id)
        _require_suspicious_review(record, req.verification_reviewed)
        preparation_fingerprint = _require_prepared_approval(req.job_id)
        session = apply_session_factory(
            req.job_id, verification_reviewed=req.verification_reviewed)
        session_id = uuid4().hex
        sessions[session_id] = session
        session_jobs[session_id] = req.job_id
        session_preparations[session_id] = preparation_fingerprint
        try:
            state = session.start()
        except ValueError as exc:
            sessions.pop(session_id, None)
            session_jobs.pop(session_id, None)
            session_preparations.pop(session_id, None)
            try:
                session.cancel()
            except Exception:
                pass
            raise HTTPException(400, str(exc)) from exc
        except SubmitBlocked as exc:
            sessions.pop(session_id, None)
            session_jobs.pop(session_id, None)
            session_preparations.pop(session_id, None)
            try:
                session.cancel()
            except Exception:
                pass
            raise HTTPException(409, str(exc)) from exc
        except Exception:
            sessions.pop(session_id, None)
            session_jobs.pop(session_id, None)
            session_preparations.pop(session_id, None)
            try:
                session.cancel()
            except Exception:
                pass
            raise
        return {"session_id": session_id, **state}

    @app.post("/api/apply/rescan")
    def apply_rescan(req: SessionRequest) -> dict:
        return {"session_id": req.session_id, **_session(req.session_id).rescan()}

    @app.post("/api/apply/edit")
    def apply_edit(req: EditRequest) -> dict:
        state = _session(req.session_id).edit(req.selector, req.value,
                                              mark_done=req.mark_done)
        return {"session_id": req.session_id, **state}

    @app.post("/api/apply/submit")
    def apply_submit(req: SubmitSessionRequest) -> dict:
        from job_agent.dashboard.apply_session import SubmitBlocked

        session = _session(req.session_id)
        try:
            current_fingerprint = _require_prepared_approval(session_jobs[req.session_id])
            if current_fingerprint != session_preparations.get(req.session_id):
                raise HTTPException(409, "Prepared application changed after this form session opened; cancel and review it again")
            result = session.submit(req.approval_token)  # tied to the reviewed state
        except SubmitBlocked as exc:
            raise HTTPException(409, str(exc)) from exc
        if (result.get("result") or {}).get("status") == "submitted":
            sessions.pop(req.session_id, None)
            session_jobs.pop(req.session_id, None)
            session_preparations.pop(req.session_id, None)
        return result

    @app.post("/api/apply/cancel")
    def apply_cancel(req: SessionRequest) -> dict:
        session = sessions.pop(req.session_id, None)
        if session is None:
            raise HTTPException(404, "no such apply session (it may have finished)")
        session_jobs.pop(req.session_id, None)
        session_preparations.pop(req.session_id, None)
        return session.cancel()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    return app
