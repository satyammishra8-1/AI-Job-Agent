"""Evidence-grounded application preparation for the local dashboard."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from job_agent.apply.answer_bank import AnswerBank
from job_agent.apply.fields import FieldType, FormField
from job_agent.apply.screening import NEEDS_INPUT_MARKER, make_drafter, make_llm_generate, verify_answer
from job_agent.json_store import read_json, write_json_atomic
from job_agent.tailor.career_facts import CareerFacts

STATES = {"NOT_READY", "RESUME_READY", "ANSWERS_READY", "NEEDS_REVIEW", "READY_TO_APPLY"}
QUESTIONS = (
    ("why_company", "Why do you want to join this company?"),
    ("fit", "Why are you a good fit for this role?"),
    ("interest", "Why are you interested in this role?"),
    ("experience", "Describe relevant experience or a project."),
    ("internship_availability", "What is your internship availability?"),
    ("graduation", "What is your expected graduation date?"),
    ("notice_availability", "What is your notice period or availability?"),
    ("technical_skills", "Which relevant technical skills do you have?"),
    ("additional", "Is there anything else you would like us to know?"),
)


def _facts_evidence(facts: CareerFacts, bank: AnswerBank) -> list[str]:
    evidence = []
    evidence.extend(f"Career fact: education {v}" for v in facts.education)
    evidence.extend(f"Career fact: education {v.degree} {v.field}; {', '.join(v.evidence)}" for v in facts.education_details)
    evidence.extend(f"Project: {v.name}; {', '.join(v.evidence)}" for v in facts.projects if v.has_evidence())
    evidence.extend(f"Experience: {v.title} at {v.company}; {', '.join(v.evidence) or ', '.join(v.real_bullets)}" for v in facts.employers)
    evidence.extend(f"Verified profile skill: {skill}" for skills in facts.skills_inventory.values() for skill in skills)
    evidence.extend(f"Verified skill evidence: {k}; {', '.join(v)}" for k, v in facts.skill_evidence.items() if v)
    evidence.extend(f"Certification: {v.name}; {', '.join(v.evidence)}" for v in facts.certifications if v.evidence)
    if facts.expected_graduation:
        evidence.append(f"Career fact: expected graduation {facts.expected_graduation}")
    if facts.internship_preference is not None:
        evidence.append(f"Career fact: internship preference {facts.internship_preference}")
    if facts.career_status:
        evidence.append(f"Career fact: career status {facts.career_status}")
    if facts.current_year:
        evidence.append(f"Career fact: current study year {facts.current_year}")
    if facts.experience_years is not None:
        evidence.append(f"Career fact: experience years {facts.experience_years:g}")
    for label, value in (("notice period", bank.notice_period),
                         ("earliest start date", bank.earliest_start_date),
                         ("availability", bank.work_mode),
                         ("location preference", bank.location_preference)):
        if value:
            evidence.append(f"Saved application profile: {label} {value}")
    return evidence


def _evidence_for(answer: str, evidence: list[str]) -> list[str]:
    """Conservatively cite profile evidence whose content appears in the answer."""
    words = set(re.findall(r"[a-z0-9+#.]+", answer.lower()))
    stop = {"career", "fact", "evidence", "project", "profile", "verified", "with", "from", "the", "and", "for"}
    cited = []
    for item in evidence:
        terms = set(re.findall(r"[a-z0-9+#.]+", item.lower())) - stop
        if terms and (terms & words):
            cited.append(item)
    return cited


def _state(path: Path) -> dict:
    data = read_json(path, {})
    if not isinstance(data, dict):
        raise ValueError(f"Invalid application preparation data in {path}")
    return data


def get_preparation(path: Path, job_id: str) -> dict:
    return _state(path).get(job_id, {"job_id": job_id, "status": "NOT_READY", "answers": []})


def generate_preparation(*, path: Path, job: dict, facts: CareerFacts, bank: AnswerBank,
                         resume_status: str, settings, jd: str,
                         resume_path: Path | None = None) -> dict:
    job_id = str(job["id"])
    if resume_status != "READY":
        raise ValueError("A validated tailored resume is required before application preparation")
    verification = (job.get("verification") or {}).get("status")
    eligibility = (job.get("eligibility") or {}).get("status")
    if verification != "verified" or eligibility != "eligible":
        raise ValueError("Only verified and eligible opportunities can be prepared")
    evidence = _facts_evidence(facts, bank)
    if not evidence and not facts.employers and not facts.skills_inventory:
        raise ValueError("Verified career evidence is required to prepare answers")
    prior = get_preparation(path, job_id)
    if prior.get("status") == "READY_TO_APPLY":
        prior["status"] = "ANSWERS_READY"
    drafter = (make_drafter(make_llm_generate(settings), facts, bank, jd=jd,
                            company=job.get("company", "")) if settings.gemini_api_key else None)
    answers = []
    for key, question in QUESTIONS:
        missing = []
        value = ""
        status = "NEEDS_REVIEW"
        reason = "No Gemini API key is configured; provide and review this answer manually." if drafter is None else ""
        if key == "graduation" and facts.expected_graduation:
            value = str(facts.expected_graduation)
        elif key == "internship_availability" and facts.internship_preference is not None:
            value = "I am interested in internship opportunities." if facts.internship_preference else "Internship availability is not stated in my profile."
            if not facts.internship_preference:
                missing.append("Confirm internship availability")
        elif key == "notice_availability" and bank.notice_period:
            value = bank.notice_period
        elif key == "technical_skills":
            supported = sorted(facts.real_skills())
            if supported:
                jd_lower = jd.lower()
                matches = [s for s in supported if s.lower() in jd_lower]
                value = ", ".join(matches or supported[:8])
                if not matches:
                    missing.append("Confirm which listed skills are relevant to this job")
        if not value and drafter is not None:
            field = FormField(selector=f"prep-{key}", field_type=FieldType.TEXTAREA,
                              label=question)
            fill = drafter(field)
            if fill:
                value = fill.value
                if NEEDS_INPUT_MARKER in value:
                    missing.append(value.split(NEEDS_INPUT_MARKER, 1)[1].strip())
                violations = verify_answer(value, facts, job.get("company", ""))
                if violations:
                    missing.extend(violations)
                reason = "Drafted against this job description and the verified career profile."
        if not value:
            missing.append("Profile does not contain enough evidence; add a truthful answer or leave blank")
        used_evidence = _evidence_for(value, evidence) if value else []
        if value and not used_evidence:
            missing.append("No specific verified profile evidence could be matched to this draft")
        answers.append({"key": key, "question": question, "answer": value,
                        "status": "NEEDS_REVIEW" if missing or not value else "ANSWERS_READY",
                        "evidence": used_evidence, "missing_evidence": missing,
                        "reasoning": reason or "Uses only explicit profile or answer-bank information."})
    status = "NEEDS_REVIEW" if any(a["status"] == "NEEDS_REVIEW" for a in answers) else "ANSWERS_READY"
    # The reusable screening verifier is also applied to generated message/note text.
    msg = next((a["answer"] for a in answers if a["key"] == "fit" and a["answer"]), "")
    if msg:
        msg = f"Hello, I’m interested in the {job.get('title', 'role')} position at {job.get('company', 'your company')}. {msg} I’d welcome the chance to discuss my application."
    cover = next((a["answer"] for a in answers if a["key"] == "interest" and a["answer"]), "")
    preparation = {"job_id": job_id, "status": status, "resume_status": resume_status,
                   "answers": answers, "recruiter_message": msg, "cover_note": cover,
                   "recruiter_message_evidence": _evidence_for(msg, evidence),
                   "cover_note_evidence": _evidence_for(cover, evidence),
                   "recruiter_message_missing_evidence": [],
                   "cover_note_missing_evidence": [],
                   "context_fingerprint": _context_fingerprint(job, resume_path),
                   "reviewed": False, "approved_fingerprint": None}
    data = _state(path); data[job_id] = preparation; write_json_atomic(path, data)
    return preparation


def update_preparation(path: Path, job_id: str, updates: dict,
                       facts: CareerFacts | None = None, company: str = "",
                       bank: AnswerBank | None = None) -> dict:
    data = _state(path); prep = data.get(job_id)
    if not isinstance(prep, dict):
        raise KeyError(job_id)
    by_key = {a["key"]: a for a in prep.get("answers", [])}
    for key, value in updates.get("answers", {}).items():
        if key not in by_key:
            raise ValueError(f"Unknown prepared answer: {key}")
        previous = by_key[key].get("answer", "")
        existing_missing = list(by_key[key].get("missing_evidence", []))
        by_key[key]["answer"] = value
        unresolved = (existing_missing if value == previous else [])
        if NEEDS_INPUT_MARKER in value and not unresolved:
            unresolved.append("Add the genuine personal information requested by the draft")
        if not value.strip():
            unresolved.append("Answer is empty")
        by_key[key]["missing_evidence"] = unresolved
        by_key[key]["status"] = "ANSWERS_READY" if value.strip() and not unresolved else "NEEDS_REVIEW"
        by_key[key]["reasoning"] = "Edited and explicitly reviewed by the user."
    for key in ("recruiter_message", "cover_note"):
        if key in updates:
            prep[key] = updates[key]
    issues = []
    if facts is not None:
        evidence = _facts_evidence(facts, bank or AnswerBank())
        for answer in by_key.values():
            label, value = answer["key"], answer.get("answer", "")
            answer["evidence"] = _evidence_for(value, evidence)
            if value and not answer["evidence"]:
                item = "No specific verified profile evidence could be matched to this answer"
                missing = list(answer.get("missing_evidence", []))
                if item not in missing:
                    missing.append(item)
                answer["missing_evidence"] = missing
                answer["status"] = "NEEDS_REVIEW"
            for issue in verify_answer(value, facts, company):
                issues.append(f"{label}: {issue}")
        for label in ("recruiter_message", "cover_note"):
            prep[f"{label}_evidence"] = _evidence_for(prep.get(label, ""), evidence)
            prep[f"{label}_missing_evidence"] = (
                [] if not prep.get(label) or prep[f"{label}_evidence"] else
                ["No specific verified profile evidence could be matched to this text"])
            for issue in verify_answer(prep.get(label, ""), facts, company):
                issues.append(f"{label}: {issue}")
    prep["content_issues"] = issues
    prep["answers"] = list(by_key.values())
    prep["reviewed"] = False; prep["approved_fingerprint"] = None
    prep["status"] = "NEEDS_REVIEW" if (issues or any(a["status"] == "NEEDS_REVIEW" for a in prep["answers"])) else "ANSWERS_READY"
    data[job_id] = prep; write_json_atomic(path, data)
    return prep


def fingerprint(prep: dict) -> str:
    approved = {k: v for k, v in prep.items() if k not in {"approved_fingerprint", "reviewed"}}
    return hashlib.sha256(json.dumps(approved, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _context_fingerprint(job: dict, resume_path: Path | None) -> str:
    state = {key: job.get(key) for key in ("id", "title", "company", "description", "url", "apply_url")}
    if resume_path and resume_path.exists():
        state["resume_sha256"] = hashlib.sha256(resume_path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def approve_preparation(path: Path, job_id: str) -> dict:
    data = _state(path); prep = data.get(job_id)
    if not isinstance(prep, dict):
        raise KeyError(job_id)
    if prep.get("status") == "NEEDS_REVIEW" or prep.get("content_issues") or any(a.get("status") != "ANSWERS_READY" for a in prep.get("answers", [])):
        raise ValueError("Resolve every NEEDS_REVIEW answer before approval")
    prep["status"] = "READY_TO_APPLY"; prep["reviewed"] = True
    prep["approved_fingerprint"] = fingerprint(prep)
    data[job_id] = prep; write_json_atomic(path, data)
    return prep


def is_approved(prep: dict, *, job: dict | None = None,
                resume_path: Path | None = None) -> bool:
    return (prep.get("status") == "READY_TO_APPLY" and prep.get("reviewed") is True
            and prep.get("approved_fingerprint") == fingerprint(prep)
            and (job is None or prep.get("context_fingerprint") ==
                 _context_fingerprint(job, resume_path)))
