from job_finder.app import AppController
from job_finder.models import CandidateProfile, JobPosting, ResumeOption, ScoredJobMatch
from job_finder.saved_jobs import SavedJobsStore
from job_finder.workspace import LocalWorkspace


def make_profile():
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


def make_match(score=9):
    return ScoredJobMatch(
        job=JobPosting(
            provider="serpapi_google_jobs",
            provider_job_id="job-1",
            title="Machine Learning Engineer",
            company="Acme AI",
            location="Montreal, QC",
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
        missing_signals=[],
    )


class FakeService:
    def __init__(self):
        self.load_calls = []
        self.search_calls = []
        self.preview_calls = []
        self.custom_resume_calls = []

    def load_rxresume_options(self, base_url, api_key):
        self.load_calls.append((base_url, api_key))
        return [
            ResumeOption(id="resume-1", label="Primary resume"),
            ResumeOption(id="resume-2", label="Fallback resume"),
        ]

    def run_search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "profile": make_profile(),
            "location_used": kwargs["location_override"] or "Toronto, ON",
            "matches": [make_match()],
            "status": "Found 1 matching job.",
        }

    def preview_profile(self, **kwargs):
        self.preview_calls.append(kwargs)
        return {
            "profile": make_profile(),
            "location_used": "Toronto, ON",
            "status": "Resume analyzed.",
            "search_terms_text": "Machine Learning Engineer\nApplied Scientist",
        }

    def generate_custom_resume(self, **kwargs):
        self.custom_resume_calls.append(kwargs)
        return {
            "status": "Custom resume and cover letter generated for Acme AI.",
            "resume_pdf_path": "/tmp/tailored_resume.pdf",
            "cover_letter_path": "/tmp/cover_letter.md",
        }

    def run_auto_apply(self, **kwargs):
        from job_finder.auto_apply import ApplyRunMode, ApplyRunResult
        from job_finder.models import GeneratedApplicationArtifacts

        if not hasattr(self, "auto_apply_calls"):
            self.auto_apply_calls = []
        self.auto_apply_calls.append(kwargs)
        artifacts = GeneratedApplicationArtifacts(
            base_resume_id=kwargs.get("rxresume_resume_id", ""),
            remote_resume_id="remote-1",
            company="Acme AI",
            job_title="Machine Learning Engineer",
            generated_at="2026-05-17T10:00:00Z",
            artifact_dir="/tmp/output/generated/run",
            pdf_url="https://example.com/pdf",
            pdf_path="/tmp/tailored_resume.pdf",
            cover_letter_path="/tmp/cover_letter.md",
            resume_json_path="/tmp/resume.json",
            metadata_path="/tmp/metadata.json",
        )
        mode_value = kwargs.get("mode")
        mode = mode_value if isinstance(mode_value, ApplyRunMode) else ApplyRunMode(str(mode_value or "attended"))
        result = ApplyRunResult(
            status="needs_review" if mode == ApplyRunMode.ATTENDED else "success",
            mode=mode,
            apply_url="https://example.com/jobs/1",
            run_id="run-abc",
            run_dir="/tmp/output/generated/run/apply_run",
            transcript_path="/tmp/output/generated/run/apply_run/transcript.json",
            screenshots_dir="/tmp/output/generated/run/apply_run/screenshots",
            started_at="2026-05-17T10:00:00Z",
            finished_at="2026-05-17T10:05:00Z",
        )
        return artifacts, result


def test_controller_loads_rxresume_options():
    service = FakeService()
    controller = AppController(service)

    status, options = controller.load_rxresume_options(
        "https://rx.example.com",
        "rx-key",
    )

    assert status == "Loaded 2 resumes."
    assert options == [("Primary resume", "resume-1"), ("Fallback resume", "resume-2")]
    assert service.load_calls == [("https://rx.example.com", "rx-key")]


def test_controller_run_search_uses_manual_location_override():
    service = FakeService()
    controller = AppController(service)

    result = controller.run_search(
        source_type="pdf",
        pdf_bytes=b"%PDF-1.7",
        pdf_filename="resume.pdf",
        rxresume_base_url="",
        rxresume_api_key="",
        rxresume_resume_id="",
        openai_api_key="sk-test",
        serpapi_api_key="serp-test",
        location_override="Montreal, QC",
        include_remote=True,
    )

    assert result["location_used"] == "Montreal, QC"
    assert result["rows"][0]["Score"] == 9
    assert service.search_calls[0]["source_type"] == "pdf"
    assert service.search_calls[0]["location_override"] == "Montreal, QC"


def test_controller_requires_resume_selection_for_rxresume_source():
    controller = AppController(FakeService())

    result = controller.run_search(
        source_type="rxresume",
        pdf_bytes=None,
        pdf_filename="",
        rxresume_base_url="https://rx.example.com",
        rxresume_api_key="rx-key",
        rxresume_resume_id="",
        openai_api_key="sk-test",
        serpapi_api_key="serp-test",
        location_override="",
        include_remote=True,
    )

    assert result["status"] == "Select a Reactive Resume entry before searching."
    assert result["rows"] == []


def test_controller_preview_profile_prefills_location_before_search():
    service = FakeService()
    controller = AppController(service)

    result = controller.preview_profile(
        source_type="pdf",
        pdf_bytes=b"%PDF-1.7",
        pdf_filename="resume.pdf",
        rxresume_base_url="",
        rxresume_api_key="",
        rxresume_resume_id="",
        openai_api_key="sk-test",
        openai_model="gpt-5",
    )

    assert result["status"] == "Resume analyzed."
    assert result["location_used"] == "Toronto, ON"
    assert "Toronto, ON" in result["profile_markdown"]
    assert result["search_terms_text"] == "Machine Learning Engineer\nApplied Scientist"
    assert service.preview_calls[0]["source_type"] == "pdf"
    assert service.preview_calls[0]["openai_model"] == "gpt-5"


def test_controller_run_search_forwards_search_terms_text():
    service = FakeService()
    controller = AppController(service)

    controller.run_search(
        source_type="pdf",
        pdf_bytes=b"%PDF-1.7",
        pdf_filename="resume.pdf",
        rxresume_base_url="",
        rxresume_api_key="",
        rxresume_resume_id="",
        openai_api_key="sk-test",
        serpapi_api_key="serp-test",
        location_override="Toronto, ON",
        include_remote=True,
        search_terms_text="Applied Scientist\nML Platform Engineer",
        openai_model="gpt-5.4",
    )

    assert service.search_calls[0]["search_terms_text"] == "Applied Scientist\nML Platform Engineer"
    assert service.search_calls[0]["openai_model"] == "gpt-5.4"


def test_controller_run_search_serializes_matches_state():
    result = AppController(FakeService()).run_search(
        source_type="pdf",
        pdf_bytes=b"%PDF-1.7",
        pdf_filename="resume.pdf",
        rxresume_base_url="",
        rxresume_api_key="",
        rxresume_resume_id="",
        openai_api_key="sk-test",
        serpapi_api_key="serp-test",
        location_override="Toronto, ON",
        include_remote=True,
    )

    assert result["matches_state"][0]["job"]["company"] == "Acme AI"


def test_controller_generate_custom_resume_forwards_payload():
    service = FakeService()
    controller = AppController(service)

    result = controller.generate_custom_resume(
        source_type="rxresume",
        rxresume_base_url="https://rxresu.me/api/openapi/resumes",
        rxresume_api_key="",
        rxresume_resume_id="resume-1",
        candidate_profile=make_profile().model_dump(),
        match=make_match().model_dump(),
        openai_api_key="sk-test",
        openai_model="gpt-5",
    )

    assert result["resume_pdf_path"] == "/tmp/tailored_resume.pdf"
    assert service.custom_resume_calls[0]["openai_model"] == "gpt-5"
    assert service.custom_resume_calls[0]["rxresume_resume_id"] == "resume-1"


def test_controller_saves_and_lists_saved_jobs(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
    )
    controller = AppController(FakeService(), workspace=workspace, saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path))

    save_result = controller.save_saved_job(make_match().model_dump())
    list_result = controller.list_saved_jobs()

    assert "Saved Machine Learning Engineer at Acme AI." == save_result["status"]
    assert save_result["saved_job_created"] is True
    assert save_result["selected_saved_job_id"] == save_result["saved_job"]["id"]
    assert save_result["rows"][0]["Company"] == "Acme AI"
    assert list_result["saved_jobs_state"][0]["id"] == save_result["saved_job"]["id"]
    assert list_result["saved_jobs_button_label"] == "Saved Jobs (1)"


def test_controller_reports_already_saved_for_duplicate_job(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
    )
    controller = AppController(FakeService(), workspace=workspace, saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path))

    first = controller.save_saved_job(make_match().model_dump())
    second = controller.save_saved_job(make_match(score=10).model_dump())

    assert first["saved_job_created"] is True
    assert second["saved_job_created"] is False
    assert second["status"] == "Already saved Machine Learning Engineer at Acme AI."
    assert len(second["rows"]) == 1
    assert second["selected_saved_job_id"] == first["saved_job"]["id"]


def test_controller_updates_and_deletes_saved_jobs(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
    )
    controller = AppController(FakeService(), workspace=workspace, saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path))
    saved = controller.save_saved_job(make_match().model_dump())["saved_job"]

    update_result = controller.update_saved_job(
        saved_job_id=saved["id"],
        provider="serpapi_google_jobs",
        provider_job_id="job-1",
        title="Senior ML Engineer",
        company="Acme AI",
        location="Toronto, ON",
        pay_range="$200k-$240k",
        via="via LinkedIn",
        description="Lead ranking systems.",
        posted_at="today",
        remote_flag=True,
        apply_url="https://example.com/jobs/1",
        share_url="https://google.com/jobs/1",
        score_10=10,
        rationale="Even stronger fit.",
        matched_skills_text="Python\nLLMs",
        missing_signals_text="",
    )
    delete_result = controller.delete_saved_job(saved["id"])

    assert update_result["saved_job"]["match"]["job"]["title"] == "Senior ML Engineer"
    assert update_result["saved_job"]["match"]["job"]["remote_flag"] is True
    assert update_result["saved_job"]["match"]["matched_skills"] == ["Python", "LLMs"]
    assert update_result["selected_saved_job_id"] == saved["id"]
    assert delete_result["deleted"] is True
    assert delete_result["rows"] == []
    assert delete_result["saved_jobs_button_label"] == "Saved Jobs (0)"


def test_controller_auto_apply_requires_full_name_and_email(tmp_path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )
    service = FakeService()
    controller = AppController(
        service,
        workspace=workspace,
        saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path),
    )

    saved = controller.save_saved_job(make_match().model_dump())["saved_job"]

    result = controller.auto_apply_saved_job(
        saved["id"],
        mode="attended",
        rxresume_base_url="https://rxresu.me/api/openapi/resumes",
        rxresume_resume_id="resume-1",
        candidate_profile=make_profile().model_dump(),
    )

    assert "full name and email" in result["status"].lower()
    assert result["apply_result"] is None
    assert not getattr(service, "auto_apply_calls", [])


def test_controller_auto_apply_saved_job_persists_status(tmp_path):
    from job_finder.models import ApplicantProfile

    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )
    workspace.save_applicant_profile(
        ApplicantProfile(full_name="Ada Lovelace", email="ada@example.com")
    )
    service = FakeService()
    store = SavedJobsStore(workspace.saved_jobs_db_path)
    controller = AppController(service, workspace=workspace, saved_jobs_store=store)

    saved = controller.save_saved_job(make_match().model_dump())["saved_job"]

    result = controller.auto_apply_saved_job(
        saved["id"],
        mode="attended",
        rxresume_base_url="https://rxresu.me/api/openapi/resumes",
        rxresume_resume_id="resume-1",
        candidate_profile=make_profile().model_dump(),
    )

    assert result["apply_result"]["status"] == "needs_review"
    assert result["apply_result"]["run_dir"].endswith("apply_run")
    assert "Auto-apply paused for review" in result["status"]

    reloaded = store.get_job(saved["id"])
    assert reloaded.application_status == "needs_review"
    assert reloaded.application_run_id == "run-abc"
    assert reloaded.application_run_path.endswith("apply_run")
    assert service.auto_apply_calls[0]["rxresume_resume_id"] == "resume-1"


def test_controller_auto_apply_requires_candidate_profile(tmp_path):
    from job_finder.models import ApplicantProfile

    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )
    workspace.save_applicant_profile(
        ApplicantProfile(full_name="Ada Lovelace", email="ada@example.com")
    )
    service = FakeService()
    controller = AppController(
        service,
        workspace=workspace,
        saved_jobs_store=SavedJobsStore(workspace.saved_jobs_db_path),
    )
    saved = controller.save_saved_job(make_match().model_dump())["saved_job"]

    result = controller.auto_apply_saved_job(
        saved["id"],
        mode="attended",
        candidate_profile=None,
    )

    assert "Analyze a resume" in result["status"]
    assert result["apply_result"] is None


def test_controller_auto_apply_persists_failed_status_on_service_exception(tmp_path):
    from job_finder.models import ApplicantProfile

    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        saved_jobs_db_path=tmp_path / "saved_jobs.sqlite3",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )
    workspace.save_applicant_profile(
        ApplicantProfile(full_name="Ada Lovelace", email="ada@example.com")
    )

    class ExplodingService(FakeService):
        def run_auto_apply(self, **kwargs):
            raise RuntimeError("boom")

    service = ExplodingService()
    store = SavedJobsStore(workspace.saved_jobs_db_path)
    controller = AppController(service, workspace=workspace, saved_jobs_store=store)
    saved = controller.save_saved_job(make_match().model_dump())["saved_job"]

    # Pre-seed an old status to confirm it gets overwritten.
    store.update_application_status(
        saved["id"], status="needs_review", run_id="old-run", run_path="/tmp/old"
    )

    result = controller.auto_apply_saved_job(
        saved["id"],
        mode="attended",
        candidate_profile=make_profile().model_dump(),
    )

    assert "Auto-apply failed" in result["status"]
    reloaded = store.get_job(saved["id"])
    assert reloaded.application_status == "failed"
    assert "boom" in reloaded.application_error
    assert reloaded.application_run_id == ""
    assert reloaded.application_run_path == ""
