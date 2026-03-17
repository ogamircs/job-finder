from types import SimpleNamespace

from job_finder.app import AppController, build_app
from job_finder.models import CandidateProfile, JobPosting, ScoredJobMatch
from job_finder.saved_jobs import SavedJobsStore
from job_finder.workspace import LocalWorkspace


def _component_by_label(app, label):
    for component in app.config["components"]:
        if component["props"].get("label") == label:
            return component
    raise AssertionError(f"Component with label {label!r} not found")


def _component_by_value(app, value):
    for component in app.config["components"]:
        if component["props"].get("value") == value:
            return component
    raise AssertionError(f"Component with value {value!r} not found")


def _component_by_elem_id(app, elem_id):
    for component in app.config["components"]:
        if component["props"].get("elem_id") == elem_id:
            return component
    raise AssertionError(f"Component with elem_id {elem_id!r} not found")


def _tab_by_label(app, label):
    for component in app.config["components"]:
        if component.get("type") == "tabitem" and component["props"].get("label") == label:
            return component
    raise AssertionError(f"Tab with label {label!r} not found")


def _first_component_by_type(app, component_type):
    for component in app.config["components"]:
        if component.get("type") == component_type:
            return component
    raise AssertionError(f"Component with type {component_type!r} not found")


def _fn_by_api_name(app, api_name):
    for index, dependency in enumerate(app.config["dependencies"]):
        if dependency["api_name"] == api_name:
            return app.fns[index].fn
    raise AssertionError(f"Dependency with api_name {api_name!r} not found")


def _make_profile():
    return CandidateProfile(
        name="Ada Lovelace",
        headline="ML Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "OpenAI"],
        industries=["Software"],
        summary_for_matching="Experienced ML engineer.",
    )


def _make_match():
    return ScoredJobMatch(
        job=JobPosting(
            provider="serpapi_google_jobs",
            provider_job_id="job-1",
            title="Machine Learning Engineer",
            company="Acme AI",
            location="Montreal, QC",
            pay_range="$180k-$220k",
            via="via LinkedIn",
            description="Build ranking systems.",
            posted_at="1 day ago",
            remote_flag=False,
            apply_url="https://example.com/jobs/1",
            share_url="https://google.com/jobs/1",
        ),
        score_10=9,
        rationale="Strong Python and ranking fit.",
        matched_skills=["Python", "OpenAI"],
        missing_signals=[],
    )


class _FakeService:
    def __init__(self):
        self.custom_resume_calls = []

    def generate_custom_resume(self, **kwargs):
        self.custom_resume_calls.append(kwargs)
        return {
            "status": "Custom resume and cover letter generated for Acme AI.",
            "resume_pdf_path": "/tmp/tailored_resume.pdf",
            "cover_letter_path": "/tmp/cover_letter.md",
        }


def test_build_app_sets_rxresume_default_endpoint():
    app = build_app()
    component = _component_by_label(app, "Reactive Resume API URL")

    assert component["props"]["value"] == "https://rxresu.me/api/openapi/resumes"


def test_build_app_starts_with_find_jobs_disabled_and_supports_tabs_and_settings():
    app = build_app()

    find_jobs_button = _component_by_value(app, "Find jobs")
    create_resume_button = _component_by_value(app, "Create custom resume and cover letter")
    _tab_by_label(app, "Job Search")
    _component_by_value(app, "Settings")
    _component_by_label(app, "Upload PDF resume")
    _component_by_label(app, "Backend model")

    assert find_jobs_button["props"]["interactive"] is False
    assert create_resume_button["props"]["interactive"] is False
    assert any(
        component.get("type") == "tabitem" and component["props"].get("label", "").startswith("Saved Jobs (")
        for component in app.config["components"]
    )


def test_build_app_adds_pay_range_to_results_table():
    app = build_app()
    dataframes = [component for component in app.config["components"] if component.get("type") == "dataframe"]

    assert any("Pay Range" in dataframe["props"]["headers"] for dataframe in dataframes)
    assert any(dataframe["props"]["headers"] == ["Score", "Title", "Company", "Location", "Pay Range", "Source", "Apply", "Saved"] for dataframe in dataframes)
    assert any(dataframe["props"]["headers"] == ["Score", "Title", "Company", "Location", "Pay Range", "Source", "Apply", "Updated"] for dataframe in dataframes)
    assert any(dataframe["props"].get("datatype", [None] * 8)[6] == "html" for dataframe in dataframes)


def test_build_app_exposes_saved_jobs_actions():
    app = build_app()

    _tab_by_label(app, "Job Search")
    _component_by_value(app, "Save job")
    _component_by_value(app, "Edit")
    _component_by_value(app, "Save changes")
    _component_by_value(app, "Delete")


def test_build_app_disables_queue_for_resume_actions():
    app = build_app()
    dependencies = {dependency["api_name"]: dependency for dependency in app.config["dependencies"]}

    assert dependencies["save_setup_ui"]["queue"] is False
    assert dependencies["preview_resume_ui"]["queue"] is True
    assert dependencies["find_jobs_ui"]["queue"] is True


def test_saved_job_resume_generation_uses_scored_job_payload(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
    )
    service = _FakeService()
    controller = AppController(
        service,
        workspace=workspace,
        saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path),
    )
    controller.save_saved_job(_make_match().model_dump())
    app = build_app(controller=controller, workspace=workspace)

    select_saved_job_ui = _fn_by_api_name(app, "select_saved_job_ui")
    create_saved_job_resume_ui = _fn_by_api_name(app, "create_saved_job_resume_ui")

    saved_jobs_state = controller.list_saved_jobs()["saved_jobs_state"]
    selection = select_saved_job_ui(saved_jobs_state, SimpleNamespace(index=0))
    selected_saved_job_id = selection[5]
    selected_saved_job_match = selection[6]

    create_saved_job_resume_ui(
        "rxresume",
        "resume-1",
        _make_profile().model_dump(),
        saved_jobs_state,
        selected_saved_job_id,
        selected_saved_job_match,
        "gpt-5",
        "https://rxresu.me/api/openapi/resumes",
    )

    forwarded_match = service.custom_resume_calls[0]["match"]
    assert forwarded_match["job"]["title"] == "Machine Learning Engineer"
    assert forwarded_match["score_10"] == 9


def test_build_app_reflects_saved_job_count_in_header(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
    )
    controller = AppController(
        _FakeService(),
        workspace=workspace,
        saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path),
    )
    controller.save_saved_job(_make_match().model_dump())

    app = build_app(controller=controller, workspace=workspace)

    _tab_by_label(app, "Saved Jobs (1)")
