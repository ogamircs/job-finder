import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from job_finder.application_documents import (
    ApplicationArtifactsService,
    apply_tailored_resume_content,
    generate_tailored_application_content,
)
from job_finder.models import (
    GeneratedApplicationArtifacts,
    JobPosting,
    ScoredJobMatch,
    TailoredApplicationContent,
    TailoredSkillCategory,
)


def make_scored_job() -> ScoredJobMatch:
    return ScoredJobMatch(
        job=JobPosting(
            provider="serpapi_google_jobs",
            provider_job_id="job-123",
            title="Senior Machine Learning Engineer",
            company="Acme AI",
            location="Toronto, ON",
            pay_range="$180k-$220k",
            via="via LinkedIn",
            description="Build recommendation systems with Python, ranking, and LLMs.",
            posted_at="2 days ago",
            remote_flag=True,
            apply_url="https://example.com/jobs/123",
            share_url="https://google.com/jobs/123",
        ),
        score_10=9,
        rationale="Strong overlap in ranking, Python, and production ML.",
        matched_skills=["Python", "LLMs", "Ranking"],
        missing_signals=["Kubernetes"],
    )


def make_tailored_content() -> TailoredApplicationContent:
    return TailoredApplicationContent(
        headline="Senior Machine Learning Engineer",
        summary="Production ML leader focused on ranking, experimentation, and applied GenAI.",
        skills=[
            TailoredSkillCategory(name="Core", keywords=["Python", "Ranking", "LLMs"]),
            TailoredSkillCategory(name="Platform", keywords=["Airflow", "Snowflake"]),
        ],
        cover_letter=(
            "Dear Hiring Team,\n\n"
            "I am excited to bring my ranking and production ML experience to Acme AI.\n"
        ),
    )


def test_apply_tailored_resume_content_updates_common_resume_fields():
    base_resume = {
        "basics": {
            "name": "Ada Lovelace",
            "label": "ML Engineer",
            "summary": "Original summary",
        },
        "summary": {"content": "<p>Original summary</p>"},
        "sections": {
            "skills": {
                "items": [
                    {"name": "Languages", "keywords": ["Python"]},
                ]
            }
        },
    }

    tailored = apply_tailored_resume_content(base_resume, make_tailored_content())

    assert tailored["basics"]["label"] == "Senior Machine Learning Engineer"
    assert tailored["basics"]["headline"] == "Senior Machine Learning Engineer"
    assert tailored["basics"]["summary"] == (
        "Production ML leader focused on ranking, experimentation, and applied GenAI."
    )
    assert "Production ML leader" in tailored["summary"]["content"]
    assert tailored["sections"]["skills"]["items"] == [
        {
            "name": "Core",
            "keywords": ["Python", "Ranking", "LLMs"],
            "id": "tailored-skill-1",
            "hidden": False,
            "icon": "",
            "proficiency": "",
            "level": 0,
        },
        {
            "name": "Platform",
            "keywords": ["Airflow", "Snowflake"],
            "id": "tailored-skill-2",
            "hidden": False,
            "icon": "",
            "proficiency": "",
            "level": 0,
        },
    ]


def test_apply_tailored_resume_content_preserves_required_skill_fields_from_template():
    base_resume = {
        "sections": {
            "skills": {
                "items": [
                    {
                        "id": "skill-1",
                        "hidden": False,
                        "icon": "brain",
                        "name": "Existing",
                        "proficiency": "expert",
                        "level": 3,
                        "keywords": ["Python"],
                    }
                ]
            }
        }
    }

    tailored = apply_tailored_resume_content(base_resume, make_tailored_content())

    assert tailored["sections"]["skills"]["items"][0] == {
        "id": "skill-1",
        "hidden": False,
        "icon": "brain",
        "name": "Core",
        "proficiency": "expert",
        "level": 3,
        "keywords": ["Python", "Ranking", "LLMs"],
    }


def test_generate_tailored_application_content_uses_requested_model():
    class FakeResponse:
        output_text = json.dumps(
            {
                "headline": "Senior Machine Learning Engineer",
                "summary": "Tailored summary",
                "skills": [{"name": "Core", "keywords": ["Python", "LLMs"]}],
                "cover_letter": "Dear Hiring Team,\n\nTailored cover letter.",
            }
        )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResponse()

    class FakeOpenAIClient:
        def __init__(self):
            self.responses = FakeResponses()

    client = FakeOpenAIClient()
    content = generate_tailored_application_content(
        base_resume={"basics": {"name": "Ada Lovelace"}},
        scored_job=make_scored_job(),
        openai_api_key="sk-test",
        openai_model="gpt-5",
        client=client,
    )

    assert content.headline == "Senior Machine Learning Engineer"
    assert client.responses.calls[0]["model"] == "gpt-5"


def test_generate_application_artifacts_writes_local_files_and_cleans_remote_resume(tmp_path: Path):
    base_resume = {
        "basics": {"name": "Ada Lovelace", "label": "ML Engineer", "summary": "Original"},
        "sections": {"skills": {"items": []}},
    }
    calls = {"delete": []}

    service = ApplicationArtifactsService(
        output_root=tmp_path,
        now_provider=lambda: datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc),
        resume_loader=lambda base_url, api_key, resume_id: base_resume,
        content_generator=lambda **kwargs: make_tailored_content(),
        resume_importer=lambda base_url, api_key, resume_payload, **kwargs: "temp-resume-123",
        pdf_exporter=lambda base_url, api_key, resume_id: "https://cdn.example.com/temp-resume-123.pdf",
        file_downloader=lambda url, destination: destination.write_bytes(b"%PDF-1.7 test"),
        resume_deleter=lambda base_url, api_key, resume_id: calls["delete"].append(
            (base_url, api_key, resume_id)
        ),
    )

    artifacts = service.generate_application_artifacts(
        rxresume_base_url="https://rx.example.com",
        rxresume_api_key="rx-key",
        base_resume_id="base-resume-1",
        scored_job=make_scored_job(),
        openai_api_key="sk-test",
        openai_model="gpt-4o",
    )

    assert isinstance(artifacts, GeneratedApplicationArtifacts)
    assert Path(artifacts.pdf_path).read_bytes() == b"%PDF-1.7 test"
    assert "Dear Hiring Team" in Path(artifacts.cover_letter_path).read_text()
    saved_resume = json.loads(Path(artifacts.resume_json_path).read_text())
    assert saved_resume["basics"]["label"] == "Senior Machine Learning Engineer"
    metadata = json.loads(Path(artifacts.metadata_path).read_text())
    assert metadata["base_resume_id"] == "base-resume-1"
    assert metadata["remote_resume_id"] == "temp-resume-123"
    assert metadata["pdf_url"] == "https://cdn.example.com/temp-resume-123.pdf"
    assert calls["delete"] == [("https://rx.example.com", "rx-key", "temp-resume-123")]


def test_generate_application_artifacts_deletes_remote_resume_on_failure(tmp_path: Path):
    calls = {"delete": []}
    service = ApplicationArtifactsService(
        output_root=tmp_path,
        resume_loader=lambda base_url, api_key, resume_id: {
            "basics": {"name": "Ada Lovelace"},
            "sections": {"skills": {"items": []}},
        },
        content_generator=lambda **kwargs: make_tailored_content(),
        resume_importer=lambda base_url, api_key, resume_payload, **kwargs: "temp-resume-999",
        pdf_exporter=lambda base_url, api_key, resume_id: "https://cdn.example.com/temp-resume-999.pdf",
        file_downloader=lambda url, destination: (_ for _ in ()).throw(RuntimeError("download failed")),
        resume_deleter=lambda base_url, api_key, resume_id: calls["delete"].append(
            (base_url, api_key, resume_id)
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        service.generate_application_artifacts(
            rxresume_base_url="https://rx.example.com",
            rxresume_api_key="rx-key",
            base_resume_id="base-resume-1",
            scored_job=make_scored_job(),
            openai_api_key="sk-test",
            openai_model="gpt-4o",
        )

    assert calls["delete"] == [("https://rx.example.com", "rx-key", "temp-resume-999")]


def test_generate_application_artifacts_returns_files_when_cleanup_delete_fails(tmp_path: Path):
    service = ApplicationArtifactsService(
        output_root=tmp_path,
        now_provider=lambda: datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc),
        resume_loader=lambda base_url, api_key, resume_id: {
            "basics": {"name": "Ada Lovelace", "label": "ML Engineer", "summary": "Original"},
            "sections": {"skills": {"items": []}},
        },
        content_generator=lambda **kwargs: make_tailored_content(),
        resume_importer=lambda base_url, api_key, resume_payload, **kwargs: "temp-resume-555",
        pdf_exporter=lambda base_url, api_key, resume_id: "https://cdn.example.com/temp-resume-555.pdf",
        file_downloader=lambda url, destination: destination.write_bytes(b"%PDF-1.7 test"),
        resume_deleter=lambda base_url, api_key, resume_id: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    artifacts = service.generate_application_artifacts(
        rxresume_base_url="https://rx.example.com",
        rxresume_api_key="rx-key",
        base_resume_id="base-resume-1",
        scored_job=make_scored_job(),
        openai_api_key="sk-test",
        openai_model="gpt-4o",
    )

    assert isinstance(artifacts, GeneratedApplicationArtifacts)
    assert Path(artifacts.pdf_path).read_bytes() == b"%PDF-1.7 test"
    assert Path(artifacts.cover_letter_path).exists()
