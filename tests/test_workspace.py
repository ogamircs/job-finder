from pathlib import Path

from job_finder.workspace import LocalWorkspace


def test_local_workspace_saves_and_loads_env_values(tmp_path: Path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
    )

    workspace.save_env_values(
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5",
            "SERPAPI_API_KEY": "serp-test",
            "RX_RESUME_API_KEY": "rx-test",
            "RX_RESUME_API_URL": "https://rxresu.me/api/openapi/resumes",
        }
    )

    assert workspace.env_exists() is True
    assert workspace.load_env_values() == {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-5",
        "SERPAPI_API_KEY": "serp-test",
        "RX_RESUME_API_KEY": "rx-test",
        "RX_RESUME_API_URL": "https://rxresu.me/api/openapi/resumes",
    }


def test_local_workspace_preserves_saved_openai_model_when_updating_other_keys(tmp_path: Path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
    )

    workspace.save_env_values(
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5.4",
            "SERPAPI_API_KEY": "serp-test",
        }
    )

    workspace.save_env_values(
        {
            "RX_RESUME_API_KEY": "rx-test",
            "RX_RESUME_API_URL": "https://rxresu.me/api/openapi/resumes",
        }
    )

    assert workspace.load_env_values() == {
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-5.4",
        "SERPAPI_API_KEY": "serp-test",
        "RX_RESUME_API_KEY": "rx-test",
        "RX_RESUME_API_URL": "https://rxresu.me/api/openapi/resumes",
    }


def test_local_workspace_saves_uploaded_resume(tmp_path: Path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
    )
    source_pdf = tmp_path / "resume.pdf"
    source_pdf.write_bytes(b"%PDF-1.7 test")

    saved = workspace.save_uploaded_resume(source_pdf)

    assert saved.name == "resume.pdf"
    assert saved.path == workspace.resume_dir / "resume.pdf"
    assert saved.path.read_bytes() == b"%PDF-1.7 test"
    assert [option.name for option in workspace.list_saved_resumes()] == ["resume.pdf"]


def test_load_applicant_profile_returns_empty_when_file_missing(tmp_path: Path):
    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )

    profile = workspace.load_applicant_profile()

    assert profile.full_name == ""
    assert profile.email == ""
    assert profile.extra_answers == {}


def test_save_and_load_applicant_profile_roundtrip(tmp_path: Path):
    from job_finder.models import ApplicantProfile

    workspace = LocalWorkspace(
        env_path=tmp_path / ".env",
        resume_dir=tmp_path / ".resume",
        applicant_profile_path=tmp_path / ".applicant_profile.json",
    )

    profile = ApplicantProfile(
        full_name="Ada Lovelace",
        email="ada@example.com",
        requires_sponsorship=True,
        salary_expectation_min=120000,
        extra_answers={"why": "Loves the team."},
    )
    workspace.save_applicant_profile(profile)

    loaded = workspace.load_applicant_profile()
    assert loaded.full_name == "Ada Lovelace"
    assert loaded.email == "ada@example.com"
    assert loaded.requires_sponsorship is True
    assert loaded.salary_expectation_min == 120000
    assert loaded.extra_answers == {"why": "Loves the team."}
