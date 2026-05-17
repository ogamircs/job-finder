from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import JobPosting, SavedJobRecord, ScoredJobMatch


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def saved_job_identity(match: ScoredJobMatch) -> str:
    provider = match.job.provider.strip().casefold()
    provider_job_id = match.job.provider_job_id.strip().casefold()
    if provider_job_id:
        return f"id:{provider}:{provider_job_id}"
    return "sig:{provider}|{title}|{company}|{apply_url}".format(
        provider=provider,
        title=match.job.title.strip().casefold(),
        company=match.job.company.strip().casefold(),
        apply_url=match.job.apply_url.strip().casefold(),
    )


def _encode_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


def _decode_list(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    parsed = json.loads(raw_value)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


@dataclass(frozen=True)
class SavedJobSaveResult:
    record: SavedJobRecord
    created: bool


class SavedJobsStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    _APPLICATION_COLUMNS: tuple[tuple[str, str], ...] = (
        ("application_status", "TEXT NOT NULL DEFAULT ''"),
        ("application_run_id", "TEXT NOT NULL DEFAULT ''"),
        ("application_run_path", "TEXT NOT NULL DEFAULT ''"),
        ("last_applied_at", "TEXT NOT NULL DEFAULT ''"),
        ("application_error", "TEXT NOT NULL DEFAULT ''"),
    )

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS saved_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    provider_job_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT '',
                    pay_range TEXT NOT NULL DEFAULT '',
                    via TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    posted_at TEXT NOT NULL DEFAULT '',
                    remote_flag INTEGER NOT NULL DEFAULT 0,
                    apply_url TEXT NOT NULL,
                    share_url TEXT NOT NULL DEFAULT '',
                    score_10 INTEGER NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    matched_skills TEXT NOT NULL DEFAULT '[]',
                    missing_signals TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    application_status TEXT NOT NULL DEFAULT '',
                    application_run_id TEXT NOT NULL DEFAULT '',
                    application_run_path TEXT NOT NULL DEFAULT '',
                    last_applied_at TEXT NOT NULL DEFAULT '',
                    application_error TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self._migrate_add_columns(connection)

    def _migrate_add_columns(self, connection: sqlite3.Connection) -> None:
        for column_name, column_decl in self._APPLICATION_COLUMNS:
            try:
                connection.execute(
                    f"ALTER TABLE saved_jobs ADD COLUMN {column_name} {column_decl}"
                )
            except sqlite3.OperationalError as exc:
                # Only swallow the duplicate-column case; surface anything else
                # (database locked, disk I/O error, malformed schema) so callers
                # see the real failure instead of a silent corruption.
                if "duplicate column" not in str(exc).casefold():
                    raise
                continue

    def _row_to_record(self, row: sqlite3.Row) -> SavedJobRecord:
        keys = set(row.keys())
        return SavedJobRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            application_status=str(row["application_status"]) if "application_status" in keys else "",
            application_run_id=str(row["application_run_id"]) if "application_run_id" in keys else "",
            application_run_path=str(row["application_run_path"]) if "application_run_path" in keys else "",
            last_applied_at=str(row["last_applied_at"]) if "last_applied_at" in keys else "",
            application_error=str(row["application_error"]) if "application_error" in keys else "",
            match=ScoredJobMatch(
                job=JobPosting(
                    provider=str(row["provider"]),
                    provider_job_id=str(row["provider_job_id"]),
                    title=str(row["title"]),
                    company=str(row["company"]),
                    location=str(row["location"]),
                    pay_range=str(row["pay_range"]),
                    via=str(row["via"]),
                    description=str(row["description"]),
                    posted_at=str(row["posted_at"]),
                    remote_flag=bool(row["remote_flag"]),
                    apply_url=str(row["apply_url"]),
                    share_url=str(row["share_url"]),
                ),
                score_10=int(row["score_10"]),
                rationale=str(row["rationale"]),
                matched_skills=_decode_list(str(row["matched_skills"])),
                missing_signals=_decode_list(str(row["missing_signals"])),
            ),
        )

    def save_match(self, match: ScoredJobMatch) -> SavedJobSaveResult:
        normalized_match = ScoredJobMatch.model_validate(match)
        dedupe_key = saved_job_identity(normalized_match)
        now = _timestamp()
        params = {
            "dedupe_key": dedupe_key,
            "provider": normalized_match.job.provider,
            "provider_job_id": normalized_match.job.provider_job_id,
            "title": normalized_match.job.title,
            "company": normalized_match.job.company,
            "location": normalized_match.job.location,
            "pay_range": normalized_match.job.pay_range,
            "via": normalized_match.job.via,
            "description": normalized_match.job.description,
            "posted_at": normalized_match.job.posted_at,
            "remote_flag": int(normalized_match.job.remote_flag),
            "apply_url": normalized_match.job.apply_url,
            "share_url": normalized_match.job.share_url,
            "score_10": normalized_match.score_10,
            "rationale": normalized_match.rationale,
            "matched_skills": _encode_list(normalized_match.matched_skills),
            "missing_signals": _encode_list(normalized_match.missing_signals),
            "created_at": now,
            "updated_at": now,
        }

        with self._connect() as connection:
            created = True
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO saved_jobs (
                        dedupe_key,
                        provider,
                        provider_job_id,
                        title,
                        company,
                        location,
                        pay_range,
                        via,
                        description,
                        posted_at,
                        remote_flag,
                        apply_url,
                        share_url,
                        score_10,
                        rationale,
                        matched_skills,
                        missing_signals,
                        created_at,
                        updated_at
                    ) VALUES (
                        :dedupe_key,
                        :provider,
                        :provider_job_id,
                        :title,
                        :company,
                        :location,
                        :pay_range,
                        :via,
                        :description,
                        :posted_at,
                        :remote_flag,
                        :apply_url,
                        :share_url,
                        :score_10,
                        :rationale,
                        :matched_skills,
                        :missing_signals,
                        :created_at,
                        :updated_at
                    )
                    """,
                    params,
                )
                row = connection.execute(
                    "SELECT * FROM saved_jobs WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
            except sqlite3.IntegrityError:
                created = False
                row = connection.execute(
                    "SELECT * FROM saved_jobs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()

        if row is None:
            raise RuntimeError("Saved job could not be loaded after write.")
        return SavedJobSaveResult(record=self._row_to_record(row), created=created)

    def list_jobs(self) -> list[SavedJobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM saved_jobs ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_job(self, saved_job_id: int) -> SavedJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM saved_jobs WHERE id = ?",
                (int(saved_job_id),),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def update_job(self, saved_job_id: int, match: ScoredJobMatch) -> SavedJobRecord | None:
        normalized_match = ScoredJobMatch.model_validate(match)
        saved_job_id = int(saved_job_id)
        dedupe_key = saved_job_identity(normalized_match)
        now = _timestamp()

        with self._connect() as connection:
            row = connection.execute(
                "SELECT created_at FROM saved_jobs WHERE id = ?",
                (saved_job_id,),
            ).fetchone()
            if row is None:
                return None
            created_at = str(row["created_at"])
            try:
                cursor = connection.execute(
                    """
                    UPDATE saved_jobs
                    SET dedupe_key = ?,
                        provider = ?,
                        provider_job_id = ?,
                        title = ?,
                        company = ?,
                        location = ?,
                        pay_range = ?,
                        via = ?,
                        description = ?,
                        posted_at = ?,
                        remote_flag = ?,
                        apply_url = ?,
                        share_url = ?,
                        score_10 = ?,
                        rationale = ?,
                        matched_skills = ?,
                        missing_signals = ?,
                        created_at = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        dedupe_key,
                        normalized_match.job.provider,
                        normalized_match.job.provider_job_id,
                        normalized_match.job.title,
                        normalized_match.job.company,
                        normalized_match.job.location,
                        normalized_match.job.pay_range,
                        normalized_match.job.via,
                        normalized_match.job.description,
                        normalized_match.job.posted_at,
                        int(normalized_match.job.remote_flag),
                        normalized_match.job.apply_url,
                        normalized_match.job.share_url,
                        normalized_match.score_10,
                        normalized_match.rationale,
                        _encode_list(normalized_match.matched_skills),
                        _encode_list(normalized_match.missing_signals),
                        created_at,
                        now,
                        saved_job_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A saved job with the same identity already exists.") from exc

            if cursor.rowcount == 0:
                return None
            updated_row = connection.execute(
                "SELECT * FROM saved_jobs WHERE id = ?",
                (saved_job_id,),
            ).fetchone()

        return self._row_to_record(updated_row) if updated_row is not None else None

    def delete_job(self, saved_job_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM saved_jobs WHERE id = ?",
                (int(saved_job_id),),
            )
        return cursor.rowcount > 0

    def update_application_status(
        self,
        saved_job_id: int,
        *,
        status: str = "",
        run_id: str = "",
        run_path: str = "",
        last_applied_at: str = "",
        error: str = "",
    ) -> SavedJobRecord | None:
        saved_job_id = int(saved_job_id)
        now = _timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE saved_jobs
                SET application_status = ?,
                    application_run_id = ?,
                    application_run_path = ?,
                    last_applied_at = ?,
                    application_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    str(status or "").strip(),
                    str(run_id or "").strip(),
                    str(run_path or "").strip(),
                    str(last_applied_at or "").strip(),
                    str(error or "").strip(),
                    now,
                    saved_job_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM saved_jobs WHERE id = ?",
                (saved_job_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None
