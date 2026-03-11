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
            "SERPAPI_API_KEY": "serp-test",
            "RX_RESUME_API_KEY": "rx-test",
            "RX_RESUME_API_URL": "https://rxresu.me/api/openapi/resumes",
        }
    )

    assert workspace.env_exists() is True
    assert workspace.load_env_values() == {
        "OPENAI_API_KEY": "sk-test",
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
