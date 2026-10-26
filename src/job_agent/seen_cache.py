"""A tiny on-disk cache of when we first saw each job.

Most boards expose a real post date, so the freshness filter uses that directly. But
some jobs (e.g. a Greenhouse posting with a null ``first_published``) have no
usable date. For those, we approximate "posted within the selected window" with
"first observed within that window": the first time we see a dateless job we record the
timestamp; on later runs we compare against it.

The cache is a small JSON file under the data dir (gitignored). It is the only
mutable-state component in the pipeline, so it's kept deliberately simple.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from job_agent.json_store import read_json, write_json_atomic


class SeenCache:
    """Maps ``"<source>:<id>"`` to the ISO timestamp we first observed it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._first_seen: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        raw = read_json(self.path, {})
        if not isinstance(raw, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
            raise ValueError(f"Invalid seen cache in {self.path}: expected string keys and values")
        try:
            for value in raw.values():
                datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp in seen cache {self.path}: {exc}") from exc
        self._first_seen = raw

    @staticmethod
    def _key(source: str, job_id: str) -> str:
        return f"{source}:{job_id}"

    def first_seen(self, source: str, job_id: str, now: datetime) -> tuple[datetime, bool]:
        """(when this job was first seen, whether THIS call recorded it).

        Records ``now`` for a never-seen job; an existing entry is returned
        untouched, so re-scans are idempotent. The boolean is the explicit
        "newly inserted" signal — callers must never infer newness by comparing
        timestamps (serialization rounding would silently zero it)."""
        key = self._key(source, job_id)
        newly = key not in self._first_seen
        if newly:
            self._first_seen[key] = now.isoformat()
        return datetime.fromisoformat(self._first_seen[key]), newly

    def observe(self, source: str, job_id: str, now: datetime) -> datetime:
        """Return when this job was first seen, recording ``now`` if it's new."""
        return self.first_seen(source, job_id, now)[0]

    def __len__(self) -> int:
        return len(self._first_seen)

    def save(self) -> None:
        """Persist the cache, creating the data dir if needed."""
        write_json_atomic(self.path, self._first_seen)
