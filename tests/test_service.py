from job_finder.app import AppController
from job_finder.models import CandidateProfile, JobPosting, ResumeOption, ScoredJobMatch


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
    assert result["rows"][0]["score"] == 9
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
