from pathlib import Path

from job_finder.models import JobPosting, SavedJobRecord, ScoredJobMatch
from job_finder.saved_jobs import SavedJobsStore


def make_match(
    *,
    provider: str = "serpapi_google_jobs",
    provider_job_id: str = "job-1",
    score: int = 9,
    title: str = "Machine Learning Engineer",
):
    return ScoredJobMatch(
        job=JobPosting(
            provider=provider,
            provider_job_id=provider_job_id,
            title=title,
            company="Acme AI",
            location="Toronto, ON",
            pay_range="$180k-$220k",
            via="via LinkedIn",
            description="Build ranking systems.",
            posted_at="1 day ago",
            remote_flag=False,
            apply_url="https://example.com/jobs/1",
            share_url="https://google.com/jobs/1",
        ),
        score_10=score,
        rationale="Strong Python and ranking fit.",
        matched_skills=["Python", "OpenAI"],
        missing_signals=["Kubernetes"],
    )


def test_saved_jobs_store_persists_and_lists_matches(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")

    save_result = store.save_match(make_match())
    record = save_result.record
    listed = store.list_jobs()

    assert save_result.created is True
    assert isinstance(record, SavedJobRecord)
    assert record.id > 0
    assert listed[0].match.job.company == "Acme AI"
    assert listed[0].match.job.pay_range == "$180k-$220k"
    assert store.get_job(record.id).match.matched_skills == ["Python", "OpenAI"]


def test_saved_jobs_store_keeps_existing_row_for_duplicate_matches(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")

    first = store.save_match(make_match(score=8))
    second = store.save_match(make_match(score=10, title="Senior Machine Learning Engineer"))

    listed = store.list_jobs()

    assert first.created is True
    assert second.created is False
    assert first.record.id == second.record.id
    assert len(listed) == 1
    assert listed[0].match.score_10 == 8
    assert listed[0].match.job.title == "Machine Learning Engineer"


def test_saved_jobs_store_persists_multiple_distinct_matches_across_reloads(tmp_path: Path):
    db_path = tmp_path / "saved_jobs.sqlite3"
    first_store = SavedJobsStore(db_path)
    first_store.save_match(make_match(provider_job_id="job-1", title="Machine Learning Engineer"))
    first_store.save_match(make_match(provider_job_id="job-2", title="Applied Scientist"))

    second_store = SavedJobsStore(db_path)
    listed = second_store.list_jobs()

    assert len(listed) == 2
    assert {record.match.job.provider_job_id for record in listed} == {"job-1", "job-2"}


def test_saved_jobs_store_distinguishes_fallback_identity_by_provider(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")

    first = store.save_match(make_match(provider="serpapi_google_jobs", provider_job_id=""))
    second = store.save_match(make_match(provider="greenhouse", provider_job_id=""))

    listed = store.list_jobs()

    assert first.created is True
    assert second.created is True
    assert first.record.id != second.record.id
    assert len(listed) == 2
    assert {record.match.job.provider for record in listed} == {"serpapi_google_jobs", "greenhouse"}


def test_saved_jobs_store_updates_and_deletes_existing_rows(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")
    record = store.save_match(make_match()).record

    updated = store.update_job(
        record.id,
        make_match(
            provider_job_id="job-99",
            score=7,
            title="Applied Scientist",
        ),
    )

    assert updated is not None
    assert updated.id == record.id
    assert updated.match.job.provider_job_id == "job-99"
    assert updated.match.job.title == "Applied Scientist"

    assert store.delete_job(record.id) is True
    assert store.get_job(record.id) is None
    assert store.list_jobs() == []


def test_saved_jobs_store_returns_missing_status_for_update_and_delete(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")

    assert store.update_job(999, make_match()) is None
    assert store.delete_job(999) is False


def test_saved_jobs_store_migrates_legacy_db_without_application_columns(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as legacy:
        legacy.executescript(
            """
            CREATE TABLE saved_jobs (
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
                updated_at TEXT NOT NULL
            );
            """
        )
        legacy.execute(
            "INSERT INTO saved_jobs (dedupe_key, provider, title, company, apply_url, score_10, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-key",
                "serpapi_google_jobs",
                "Legacy Role",
                "Legacy Co",
                "https://example.com/legacy",
                7,
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )

    # Instantiating the store should ALTER TABLE in the new columns idempotently.
    store = SavedJobsStore(db_path)
    SavedJobsStore(db_path)  # second instantiation must not raise

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(saved_jobs)").fetchall()}
    assert {
        "application_status",
        "application_run_id",
        "application_run_path",
        "last_applied_at",
        "application_error",
    }.issubset(columns)

    records = store.list_jobs()
    assert len(records) == 1
    assert records[0].application_status == ""


def test_update_application_status_roundtrips(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")
    record = store.save_match(make_match()).record

    updated = store.update_application_status(
        record.id,
        status="needs_review",
        run_id="run-abc123",
        run_path="/tmp/output/run",
        last_applied_at="2026-05-17T10:00:00Z",
        error="",
    )

    assert updated is not None
    assert updated.application_status == "needs_review"
    assert updated.application_run_id == "run-abc123"
    assert updated.application_run_path == "/tmp/output/run"
    assert updated.last_applied_at == "2026-05-17T10:00:00Z"

    reloaded = store.get_job(record.id)
    assert reloaded.application_status == "needs_review"


def test_update_application_status_returns_none_when_missing(tmp_path: Path):
    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")
    assert store.update_application_status(404, status="failed") is None


def test_migrate_add_columns_reraises_non_duplicate_operational_errors(tmp_path: Path):
    import sqlite3
    import pytest

    store = SavedJobsStore(tmp_path / "saved_jobs.sqlite3")

    class _BoomConnection:
        def execute(self, sql, *args, **kwargs):
            if "ALTER TABLE" in sql:
                raise sqlite3.OperationalError("database is locked")
            return None

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store._migrate_add_columns(_BoomConnection())
