from __future__ import annotations

import asyncio
import json
import re
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from .models import (
    ApplicantProfile,
    GeneratedApplicationArtifacts,
    ScoredJobMatch,
)


class ApplyRunMode(str, Enum):
    ATTENDED = "attended"
    AUTO_SUBMIT = "auto_submit"


class ApplyRunResult(BaseModel):
    status: str = "failed"
    mode: ApplyRunMode = ApplyRunMode.ATTENDED
    apply_url: str = ""
    final_url: str = ""
    transcript_path: str = ""
    screenshots_dir: str = ""
    run_id: str = ""
    run_dir: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "status",
        "apply_url",
        "final_url",
        "transcript_path",
        "screenshots_dir",
        "run_id",
        "run_dir",
        "error",
        "started_at",
        "finished_at",
        mode="before",
    )
    @classmethod
    def _clean_apply_result_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("notes", mode="before")
    @classmethod
    def _clean_notes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        cleaned: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
        return cleaned


_BLANK_PROFILE_KEYS: tuple[str, ...] = (
    "full_name",
    "email",
    "phone",
    "location_city",
    "location_region",
    "location_country",
    "postal_code",
    "work_authorization",
    "linkedin_url",
    "github_url",
    "portfolio_url",
    "preferred_pronouns",
    "desired_start_date",
    "default_cover_letter_signoff",
)


def _profile_lines(profile: ApplicantProfile) -> list[str]:
    lines: list[str] = []
    for key in _BLANK_PROFILE_KEYS:
        value = getattr(profile, key, "")
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")

    if profile.requires_sponsorship:
        lines.append("- requires_sponsorship: true")
    if profile.willing_to_relocate:
        lines.append("- willing_to_relocate: true")

    if profile.salary_expectation_min is not None or profile.salary_expectation_max is not None:
        low = profile.salary_expectation_min
        high = profile.salary_expectation_max
        currency = profile.salary_currency or "USD"
        if low is not None and high is not None:
            lines.append(f"- salary_expectation: {low}-{high} {currency}")
        elif low is not None:
            lines.append(f"- salary_expectation_min: {low} {currency}")
        elif high is not None:
            lines.append(f"- salary_expectation_max: {high} {currency}")

    if profile.years_experience_override is not None:
        lines.append(f"- years_experience: {profile.years_experience_override}")
    if profile.notice_period_weeks is not None:
        lines.append(f"- notice_period_weeks: {profile.notice_period_weeks}")

    for key, value in (profile.extra_answers or {}).items():
        if value.strip():
            lines.append(f"- extra/{key}: {value.strip()}")
    for key, value in (profile.demographics or {}).items():
        if value.strip():
            lines.append(f"- demographics/{key}: {value.strip()}")

    return lines


def build_apply_task_prompt(
    *,
    match: ScoredJobMatch,
    artifacts: GeneratedApplicationArtifacts,
    profile: ApplicantProfile,
    mode: ApplyRunMode,
) -> str:
    """Build the task prompt for browser_use.Agent.

    Patterned after browser-use/examples/use-cases/apply_to_job.py — numbered
    step plan, explicit action types (input_text / click / upload_file_to_element),
    and a final_result contract.
    """

    job = match.job
    pdf_path = Path(artifacts.pdf_path).resolve()
    cover_letter_path = Path(artifacts.cover_letter_path).resolve()

    profile_block = "\n".join(_profile_lines(profile)) or "- (no applicant info provided)"
    if mode == ApplyRunMode.ATTENDED:
        submit_rule = (
            "DO NOT click the final Submit button. After every field is filled, "
            "use the done action with a final_result describing the form state so "
            "the human reviewer can verify the entries and click Submit themselves."
        )
    else:
        submit_rule = (
            "Once every field is filled, click the final Submit button and confirm "
            "the success screen before calling the done action."
        )

    return f"""
You are a job-application assistant driving a real browser to apply for a job.

JOB
- title: {job.title}
- company: {job.company}
- location: {job.location or 'unspecified'}
- apply_url: {job.apply_url}

LOCAL FILES (registered as available_file_paths — use upload_file_to_element with this exact path)
- tailored_resume_pdf: {pdf_path}
- cover_letter_markdown: {cover_letter_path}

APPLICANT PROFILE (source of truth — never invent answers not present here)
{profile_block}

INSTRUCTIONS
- Navigate to apply_url. If a login wall, captcha, SSO screen, or any modal blocks
  the form and you cannot proceed, close it; if you still cannot proceed, STOP and
  use the done action with a final_result that describes the blocker.
- Use the extract_structured_data action first to scan the entire application form
  and produce a step-by-step plan listing every field. Refer back to the applicant
  profile above whenever you fill a field.
- Fill the form from top to bottom. Do not skip any field — even optional ones —
  unless the answer is genuinely unknown from the applicant profile. One field per
  step.
- For text fields, use the input_text action. For dropdowns/radios/checkboxes use
  the click action. For resume/CV upload fields use the upload_file_to_element
  action with the tailored_resume_pdf path above.
- If a free-text cover-letter field appears, paste the contents of the cover_letter_markdown file.
- Skip optional EEOC / demographic questions unless the applicant profile
  explicitly provides answers in the demographics map.
- Never fabricate employment history, education, certifications, or references not
  present in the profile or resume.

MODE RULE
{submit_rule}

FINAL RESULT
At the end of the task, call the done action with a final_result containing:
  1. a plain-language summary of every detection and action you performed,
  2. a list of every question/field encountered in the form,
  3. the page URL you ended on,
  4. an explicit status string — one of: "success" (submitted), "needs_review"
     (filled but not submitted, e.g. attended mode), "failed" (blocker).
""".strip()


def default_browser_agent_factory(
    *,
    openai_api_key: str,
    openai_model: str,
    headless: bool,
    output_dir: Path,
    available_file_paths: list[str] | None = None,
) -> Callable[[str], Any]:
    """Return a callable that runs a browser_use.Agent over the given task prompt.

    Mirrors the upstream apply_to_job example
    (browser-use/examples/use-cases/apply_to_job.py):
        from browser_use import Agent, Browser, ChatOpenAI, Tools
        from browser_use.tools.views import UploadFileAction
        llm = ChatOpenAI(model=...)
        browser = Browser(cross_origin_iframes=True)
        tools = Tools(); @tools.action(...) async def upload_resume(...): ...
        agent = Agent(task=..., llm=llm, browser=browser, tools=tools,
                      available_file_paths=[resume_path])
        history = await agent.run(); history.final_result()

    Lazy-imports browser_use so the module loads without the dependency
    installed (tests inject a fake factory).
    """

    file_paths = list(available_file_paths or [])
    resume_path = file_paths[0] if file_paths else ""

    def _run(task_prompt: str) -> Any:
        from browser_use import Agent, Browser, ChatOpenAI, Tools  # type: ignore[import-not-found]
        from browser_use.tools.views import UploadFileAction  # type: ignore[import-not-found]

        llm = ChatOpenAI(model=openai_model, api_key=openai_api_key)
        browser = Browser(cross_origin_iframes=True, headless=headless)
        transcript_path = output_dir / "transcript.json"

        tools = Tools()

        if resume_path:
            @tools.action(description="Upload the tailored resume PDF to the focused upload field")
            async def upload_resume(browser_session):  # noqa: ANN001 - browser_use injects session
                # browser_use custom actions must return str | ActionResult | None.
                # UploadFileAction is the *param* model for the built-in upload_file
                # action, not a valid return value. Mirror the upstream example
                # (browser-use/examples/use-cases/apply_to_job.py:43) which returns
                # a status string while the agent itself invokes the built-in
                # upload_file_to_element action with the resume path advertised in
                # available_file_paths.
                _ = UploadFileAction  # imported solely for upstream parity / type ref
                return f"Resume PDF ready at {resume_path} — call upload_file_to_element with this exact path."

        agent = Agent(
            task=task_prompt,
            llm=llm,
            browser=browser,
            tools=tools,
            use_vision=True,
            available_file_paths=file_paths,
            save_conversation_path=str(transcript_path),
        )

        async def _drive() -> Any:
            history = await agent.run()
            final = None
            if hasattr(history, "final_result"):
                try:
                    final = history.final_result()
                except Exception:
                    final = None
            return {"history": history, "final_result": final}

        return asyncio.run(_drive())

    return _run


def _now_iso(now_provider: Callable[[], datetime]) -> str:
    return now_provider().isoformat(timespec="seconds").replace("+00:00", "Z")


def _try_parse_status_json(raw_text: str) -> dict[str, Any]:
    """Parse the final_result string when the agent emits JSON per prompt contract."""

    text = (raw_text or "").strip()
    if not text:
        return {}
    candidates: list[str] = [text]
    # Allow JSON embedded in surrounding prose (the prompt asks for a JSON tail).
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_final_payload(agent_result: Any) -> dict[str, Any]:
    """Normalize whatever the factory returned into a flat dict for inspection.

    The default factory returns ``{"history": history, "final_result": final}``.
    Fake factories in tests may return a plain dict or an object with attributes.
    """

    if isinstance(agent_result, dict):
        final = agent_result.get("final_result")
        flat = dict(agent_result)
        if isinstance(final, dict):
            for key, value in final.items():
                flat.setdefault(key, value)
        elif isinstance(final, str):
            flat.setdefault("summary", final)
            parsed = _try_parse_status_json(final)
            for key, value in parsed.items():
                flat.setdefault(key, value)
        return flat
    if agent_result is None:
        return {}
    return {"final_result": agent_result}


def _extract_final_url(payload: dict[str, Any]) -> str:
    for key in ("final_url", "current_url", "url", "page_url"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return ""


_NEGATED_SUBMIT_PATTERNS = (
    re.compile(r"\bnot\s+submitted\b"),
    re.compile(r"\bdid\s+not\s+submit\b"),
    re.compile(r"\bfailed\s+to\s+submit\b"),
    re.compile(r"\bcould\s+not\s+submit\b"),
    re.compile(r"\bunable\s+to\s+submit\b"),
    re.compile(r"\bsubmission\s+failed\b"),
    re.compile(r"\bsubmit\s+failed\b"),
    re.compile(r"\bno\s+success\s+screen\b"),
    re.compile(r"\bnever\s+submitted\b"),
)

_SUCCESS_CUE_PATTERNS = (
    re.compile(r"\bapplication\s+submitted\b"),
    re.compile(r"\bsubmitted\s+successfully\b"),
    re.compile(r"\bsuccessfully\s+submitted\b"),
    re.compile(r"\bsuccessfully\s+applied\b"),
    re.compile(r"\bapplication\s+sent\b"),
    re.compile(r"\bconfirmation\s+(?:page|screen)\s+shown\b"),
    re.compile(r"\bsuccess\s+screen\b"),
)


def _extract_status(payload: dict[str, Any], mode: ApplyRunMode) -> str:
    indicated = str(payload.get("status") or "").strip().casefold()
    summary = str(payload.get("summary") or payload.get("final_result") or "").casefold()

    # Hard-blocker keywords always win, regardless of an optimistic status field.
    if "captcha" in summary or "blocker" in summary or "login wall" in summary:
        return "failed"

    # If the summary contradicts an optimistic "submitted"/"success" status
    # field (e.g. payload claims "submitted" but the agent's prose says
    # "never submitted" / "failed to submit"), the negation wins.
    summary_negated = any(pattern.search(summary) for pattern in _NEGATED_SUBMIT_PATTERNS)
    if summary_negated:
        return "failed"

    if indicated in {"success", "submitted", "completed"}:
        return "success" if mode == ApplyRunMode.AUTO_SUBMIT else "needs_review"
    if indicated in {"needs_review", "paused", "stopped"}:
        return "needs_review"
    if indicated in {"failed", "error"}:
        return "failed"

    if any(pattern.search(summary) for pattern in _SUCCESS_CUE_PATTERNS):
        return "success" if mode == ApplyRunMode.AUTO_SUBMIT else "needs_review"

    # No explicit success signal — default to needs_review for BOTH modes so we
    # never falsely record an unverified auto-submit as success.
    return "needs_review"


class AutoApplyRunner:
    def __init__(
        self,
        *,
        browser_agent_factory: Callable[..., Callable[[str], Any]] | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.browser_agent_factory = browser_agent_factory or default_browser_agent_factory
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        *,
        match: ScoredJobMatch,
        artifacts: GeneratedApplicationArtifacts,
        profile: ApplicantProfile,
        mode: ApplyRunMode,
        openai_api_key: str,
        openai_model: str,
        output_dir: Path | str,
    ) -> ApplyRunResult:
        mode = ApplyRunMode(mode) if not isinstance(mode, ApplyRunMode) else mode
        run_id = uuid.uuid4().hex[:12]
        run_dir = Path(output_dir) / "apply_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir = run_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = run_dir / "transcript.json"

        started_at = _now_iso(self.now_provider)
        prompt = build_apply_task_prompt(
            match=match,
            artifacts=artifacts,
            profile=profile,
            mode=mode,
        )

        result_kwargs: dict[str, Any] = {
            "mode": mode,
            "apply_url": match.job.apply_url,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "transcript_path": str(transcript_path),
            "screenshots_dir": str(screenshots_dir),
            "started_at": started_at,
        }

        available_file_paths = [
            str(Path(artifacts.pdf_path).resolve()),
            str(Path(artifacts.cover_letter_path).resolve()),
        ]

        try:
            factory = self.browser_agent_factory(
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                headless=(mode == ApplyRunMode.AUTO_SUBMIT),
                output_dir=run_dir,
                available_file_paths=available_file_paths,
            )
            agent_result = factory(prompt)
            payload = _coerce_final_payload(agent_result)
            status = _extract_status(payload, mode)
            final_url = _extract_final_url(payload)
            result = ApplyRunResult(
                status=status,
                final_url=final_url,
                **result_kwargs,
            )
        except Exception as exc:
            result = ApplyRunResult(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                **result_kwargs,
            )
            try:
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass

        result = result.model_copy(update={"finished_at": _now_iso(self.now_provider)})
        try:
            (run_dir / "apply_result.json").write_text(
                json.dumps(result.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            if not result.error:
                result = result.model_copy(
                    update={
                        "status": "failed",
                        "error": f"Failed to write apply_result.json: {exc}",
                    }
                )
        return result
