"""Persist a search run so `tailor --job <id>` can pick a job up later.

Written to ``data/last_search.json`` (gitignored). Each record is the normalized
Job plus its board token and score/verdict, keyed by job id.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from job_agent.models import ScoredJob
from job_agent.json_store import read_json, write_json_atomic


def save_search(scored: list[ScoredJob], boards: list[str], path: str | Path, *,
                first_seen: dict[str, str] | None = None,
                new_job_ids: tuple[str, ...] = (),
                baseline: bool = False,
                sources_queried: int | None = None) -> Path:
    """Persist scored jobs (parallel to ``boards``) keyed by id, plus scan
    metadata for the dashboard.

    ``first_seen_this_scan`` is an EXPLICIT persisted boolean per record (from
    the seen cache's newly-inserted signal) — never derived from timestamp
    equality. Baseline rule: when the seen cache was empty at scan start there
    is no earlier scan to be "new since", so no record is badged and
    ``new_count`` is stored as null, not zero. An unknown ``sources_queried``
    is omitted entirely (no made-up numbers).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_set = set() if baseline else {str(i) for i in new_job_ids}
    records: dict[str, dict] = {}
    for s, board in zip(scored, boards):
        job_id = str(s.job.id)
        rec = s.job.model_dump(mode="json")
        rec.update(board=board, score=s.score, verdict=s.verdict, reasons=list(s.reasons),
                   first_seen_this_scan=job_id in new_set)
        if first_seen and job_id in first_seen:
            rec["first_seen"] = first_seen[job_id]
        records[job_id] = rec
    meta: dict = {
        "total": len(records),
        "new_count": (None if baseline else
                      sum(1 for r in records.values() if r["first_seen_this_scan"])),
    }
    if sources_queried is not None:
        meta["sources_queried"] = sources_queried
    write_json_atomic(path,
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "meta": meta, "jobs": records},
    )
    return path


def load_job_record(path: str | Path, job_id: str) -> dict | None:
    """Return the stored record for ``job_id``, or None if absent."""
    path = Path(path)
    if not path.exists():
        return None
    data = read_json(path, {})
    if not isinstance(data, dict) or not isinstance(data.get("jobs", {}), dict):
        raise ValueError(f"Invalid search data in {path}: expected an object with jobs")
    return data.get("jobs", {}).get(str(job_id))


def resolve_apply_url(record: dict) -> str | None:
    """The URL an apply flow must open for ``record``: its stored ``apply_url``,
    else its ``url``, VERBATIM — never a URL derived from the company, board, or
    a template. Query strings matter (SmartRecruiters' ``?oga=true`` routes to
    the per-job apply flow); any rewrite lands on the wrong page."""
    candidate = record.get("apply_url") or record.get("url") or None
    return validate_application_url(candidate) if candidate else None


def validate_application_url(url: str) -> str:
    """Allow public HTTP(S) URLs only; reject local and reserved destinations."""
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("Application URL is empty or malformed")
    if "\\" in url or any(ord(char) < 0x20 or ord(char) == 0x7F for char in url):
        raise ValueError("Application URL contains unsafe characters")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"Application URL is malformed: {exc}") from exc
    if parts.scheme.lower() not in {"http", "https"} or not host:
        raise ValueError("Application URL must use http or https and include a hostname")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Application URL must not contain embedded credentials")
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Application URL cannot target a local hostname")
    if "%" in host or re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*", host):
        # Browsers accept shorthand/hex IPv4 forms which ipaddress does not.
        raise ValueError("Application URL cannot use an encoded or ambiguous IP hostname")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Application URL cannot target a private or reserved IP address")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Application URL has an invalid port")
    return url
