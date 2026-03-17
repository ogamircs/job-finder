from pathlib import Path

from job_finder.models import JobPosting, SavedJobRecord, ScoredJobMatch
from job_finder.saved_jobs import SavedJobsStore


def make_match(*, provider_job_id: str = "job-1", score: int = 9, title: str = "Machine Learning Engineer"):
    return ScoredJobMatch(
        job=JobPosting(
            provider="serpapi_google_jobs",
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
