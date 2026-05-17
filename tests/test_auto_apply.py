from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from job_finder.auto_apply import (
    ApplyRunMode,
    ApplyRunResult,
    AutoApplyRunner,
    build_apply_task_prompt,
)
from job_finder.models import (
    ApplicantProfile,
    GeneratedApplicationArtifacts,
    JobPosting,
    ScoredJobMatch,
)


def make_match() -> ScoredJobMatch:
    return ScoredJobMatch(
        job=JobPosting(
            provider="serpapi_google_jobs",
            provider_job_id="job-1",
            title="Machine Learning Engineer",
            company="Acme AI",
            location="Toronto, ON",
            via="via Acme Careers",
            description="Build ranking systems.",
            apply_url="https://careers.acme.ai/jobs/1",
            share_url="https://google.com/jobs/1",
        ),
        score_10=9,
        rationale="Strong Python fit.",
        matched_skills=["Python"],
        missing_signals=[],
    )


def make_artifacts(tmp_path: Path) -> GeneratedApplicationArtifacts:
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    cover_letter_path = tmp_path / "cover_letter.md"
    cover_letter_path.write_text("Hello hiring manager.", encoding="utf-8")
    return GeneratedApplicationArtifacts(
        base_resume_id="base-1",
        remote_resume_id="remote-1",
        company="Acme AI",
        job_title="Machine Learning Engineer",
        generated_at="2026-05-17T10:00:00Z",
        artifact_dir=str(tmp_path),
        pdf_url="https://example.com/pdf",
        pdf_path=str(pdf_path),
        cover_letter_path=str(cover_letter_path),
        resume_json_path=str(tmp_path / "resume.json"),
        metadata_path=str(tmp_path / "metadata.json"),
    )


def make_profile(**overrides):
    base = dict(
        full_name="Ada Lovelace",
        email="ada@example.com",
        phone="555-0100",
        location_city="Toronto",
        work_authorization="Canadian Citizen",
        willing_to_relocate=True,
        salary_expectation_min=120000,
        salary_expectation_max=150000,
        linkedin_url="https://linkedin.com/in/ada",
        extra_answers={"why_acme": "Love your ranking research."},
    )
    base.update(overrides)
    return ApplicantProfile(**base)


def test_applicant_profile_cleans_and_coerces():
    profile = ApplicantProfile.model_validate(
        {
            "full_name": "  Ada Lovelace  ",
            "email": "ada@example.com",
            "requires_sponsorship": "yes",
            "willing_to_relocate": "false",
            "salary_expectation_min": "120000",
            "salary_expectation_max": "",
            "years_experience_override": "7.5",
            "notice_period_weeks": " 4 ",
            "extra_answers": {"why": " Love it "},
            "demographics": None,
        }
    )

    assert profile.full_name == "Ada Lovelace"
    assert profile.requires_sponsorship is True
    assert profile.willing_to_relocate is False
    assert profile.salary_expectation_min == 120000
    assert profile.salary_expectation_max is None
    assert profile.years_experience_override == 7.5
    assert profile.notice_period_weeks == 4
    assert profile.extra_answers == {"why": "Love it"}
    assert profile.demographics == {}


def test_build_apply_task_prompt_includes_resume_path_and_skips_blanks(tmp_path):
    artifacts = make_artifacts(tmp_path)
    profile = make_profile(github_url="", portfolio_url="")
    prompt = build_apply_task_prompt(
        match=make_match(),
        artifacts=artifacts,
        profile=profile,
        mode=ApplyRunMode.ATTENDED,
    )

    assert str(Path(artifacts.pdf_path).resolve()) in prompt
    assert "DO NOT click the final Submit button" in prompt
    assert "Ada Lovelace" in prompt
    assert "github_url" not in prompt  # blank skipped
    assert "extra/why_acme: Love your ranking research." in prompt


def test_build_apply_task_prompt_auto_submit_rule(tmp_path):
    prompt = build_apply_task_prompt(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=ApplyRunMode.AUTO_SUBMIT,
    )

    assert "click the final Submit button" in prompt
    assert "DO NOT click" not in prompt


def test_runner_returns_failed_when_factory_raises(tmp_path):
    captured = {}

    def factory(**factory_kwargs):
        captured["factory_kwargs"] = factory_kwargs

        def _run(_prompt: str):
            raise RuntimeError("boom")

        return _run

    runner = AutoApplyRunner(
        browser_agent_factory=factory,
        now_provider=lambda: datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc),
    )
    artifacts = make_artifacts(tmp_path)

    result = runner.run(
        match=make_match(),
        artifacts=artifacts,
        profile=make_profile(),
        mode=ApplyRunMode.ATTENDED,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )

    assert result.status == "failed"
    assert "boom" in result.error
    apply_result_path = tmp_path / "apply_run" / "apply_result.json"
    assert apply_result_path.exists()
    sidecar = json.loads(apply_result_path.read_text())
    assert sidecar["status"] == "failed"
    assert captured["factory_kwargs"]["headless"] is False
    assert captured["factory_kwargs"]["openai_model"] == "gpt-4o"
    available = captured["factory_kwargs"]["available_file_paths"]
    assert str(Path(artifacts.pdf_path).resolve()) in available


def test_runner_attended_mode_maps_to_needs_review(tmp_path):
    def factory(**_factory_kwargs):
        def _run(_prompt: str):
            return {"final_result": "Filled all fields, awaiting human submit.", "final_url": "https://careers.acme.ai/jobs/1/review"}

        return _run

    runner = AutoApplyRunner(
        browser_agent_factory=factory,
        now_provider=lambda: datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc),
    )

    result = runner.run(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=ApplyRunMode.ATTENDED,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )

    assert result.status == "needs_review"
    assert result.final_url == "https://careers.acme.ai/jobs/1/review"
    assert isinstance(result, ApplyRunResult)


def test_runner_auto_submit_records_success_and_sidecar(tmp_path):
    def factory(**_factory_kwargs):
        def _run(_prompt: str):
            return {"status": "submitted", "final_url": "https://careers.acme.ai/done"}

        return _run

    now = datetime(2026, 5, 17, 10, 0, 0, tzinfo=timezone.utc)
    runner = AutoApplyRunner(
        browser_agent_factory=factory,
        now_provider=lambda: now,
    )

    result = runner.run(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=ApplyRunMode.AUTO_SUBMIT,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )

    assert result.status == "success"
    sidecar = json.loads((tmp_path / "apply_run" / "apply_result.json").read_text())
    assert sidecar["status"] == "success"
    assert sidecar["final_url"] == "https://careers.acme.ai/done"


def test_runner_auto_submit_without_explicit_signal_defaults_to_needs_review(tmp_path):
    """Regression: auto_submit must NOT optimistically record success when
    Browser Use returned no explicit success indicator."""

    def factory(**_factory_kwargs):
        def _run(_prompt: str):
            return {"final_result": "Walked through the form."}  # no status, no submit keyword

        return _run

    runner = AutoApplyRunner(browser_agent_factory=factory)
    result = runner.run(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=ApplyRunMode.AUTO_SUBMIT,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )

    assert result.status == "needs_review"


def test_runner_parses_status_json_embedded_in_final_result_string(tmp_path):
    def factory(**_factory_kwargs):
        def _run(_prompt: str):
            return {
                "final_result": 'Summary first. {"status":"failed","final_url":"https://example.com/x"} trailing prose.'
            }

        return _run

    runner = AutoApplyRunner(browser_agent_factory=factory)
    result = runner.run(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=ApplyRunMode.AUTO_SUBMIT,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )

    assert result.status == "failed"
    assert result.final_url == "https://example.com/x"


@pytest.mark.parametrize(
    "mode_value", ["attended", ApplyRunMode.ATTENDED]
)
def test_runner_accepts_mode_as_str_or_enum(tmp_path, mode_value):
    def factory(**_factory_kwargs):
        def _run(_prompt: str):
            return {"status": "needs_review"}

        return _run

    runner = AutoApplyRunner(browser_agent_factory=factory)
    result = runner.run(
        match=make_match(),
        artifacts=make_artifacts(tmp_path),
        profile=make_profile(),
        mode=mode_value,
        openai_api_key="sk-test",
        openai_model="gpt-4o",
        output_dir=tmp_path,
    )
    assert result.mode == ApplyRunMode.ATTENDED
