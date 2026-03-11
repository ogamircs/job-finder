from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd
from openai import OpenAI

from .job_provider import SerpApiGoogleJobsProvider
from .matching import build_search_queries, find_job_matches
from .models import CandidateProfile, ResumeOption, SearchRequest, SearchRunResult
from .resume_sources import (
    DEFAULT_RXRESUME_RESUMES_URL,
    ensure_candidate_profile_has_signal,
    list_rxresume_resumes,
    load_candidate_profile_from_pdf,
    load_candidate_profile_from_rxresume,
)
from .workspace import LocalWorkspace

APP_CSS = """
#search-header {
    align-items: flex-start;
}

#search-settings-toggle {
    justify-content: flex-end;
}

#settings-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
}

#job-results-table {
    overflow-x: auto;
}

#job-results-table .table-wrap,
#job-results-table .table-container,
#job-results-table .wrap,
#job-results-table > .wrap {
    overflow-x: auto !important;
}

#job-results-table table {
    min-width: 1480px;
}
"""


def _rows_from_matches(matches: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        rows.append(
            {
                "score": match.score_10,
                "title": match.job.title,
                "company": match.job.company,
                "location": match.job.location or ("Remote" if match.job.remote_flag else ""),
                "pay_range": match.job.pay_range,
                "via": match.job.via,
                "apply_url": match.job.apply_url,
                "rationale": match.rationale,
            }
        )
    return rows


def _profile_markdown(profile: CandidateProfile, *, location_used: str) -> str:
    lines = [
        f"**Candidate:** {profile.name or 'Unknown'}",
        f"**Headline:** {profile.headline or 'Not provided'}",
        f"**Location used:** {location_used or 'Remote only'}",
    ]
    if profile.target_titles:
        lines.append(f"**Target titles:** {', '.join(profile.target_titles)}")
    if profile.top_skills:
        lines.append(f"**Top skills:** {', '.join(profile.top_skills[:8])}")
    return "\n\n".join(lines)


def _results_markdown(matches: list[Any]) -> str:
    if not matches:
        return "No jobs met the 7/10 threshold."
    lines: list[str] = []
    for match in matches:
        pay_text = f" | Pay: {match.job.pay_range}" if match.job.pay_range else ""
        lines.append(
            f"- **{match.job.title}** at **{match.job.company}** in "
            f"{match.job.location or ('Remote' if match.job.remote_flag else 'Unspecified')} "
            f"({match.score_10}/10){pay_text} - [Apply]({match.job.apply_url})"
        )
    return "\n".join(lines)


def _parse_search_terms_text(search_terms_text: str) -> list[str]:
    seen: set[str] = set()
    search_terms: list[str] = []
    for line in search_terms_text.splitlines():
        for piece in line.split(","):
            cleaned = piece.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            search_terms.append(cleaned)
            if len(search_terms) >= 3:
                return search_terms
    return search_terms


def _search_terms_text_for_profile(profile: CandidateProfile) -> str:
    return "\n".join(build_search_queries(profile))


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["score", "title", "company", "location", "pay_range", "via", "apply_url", "rationale"]
    )


class JobMatchService:
    def __init__(
        self,
        *,
        pdf_loader=load_candidate_profile_from_pdf,
        rxresume_options_loader=list_rxresume_resumes,
        rxresume_profile_loader=load_candidate_profile_from_rxresume,
        provider_factory=None,
        matcher=find_job_matches,
        workspace: LocalWorkspace | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader
        self.rxresume_options_loader = rxresume_options_loader
        self.rxresume_profile_loader = rxresume_profile_loader
        self.provider_factory = provider_factory or (lambda api_key: SerpApiGoogleJobsProvider(api_key=api_key))
        self.matcher = matcher
        self.workspace = workspace or LocalWorkspace()

    def _resolve_secret(self, env_var: str, value: str = "") -> str:
        return self.workspace.resolve_value(env_var, value)

    def _resolve_rxresume_base_url(self, value: str = "") -> str:
        return self.workspace.resolve_value("RX_RESUME_API_URL", value) or DEFAULT_RXRESUME_RESUMES_URL

    def load_rxresume_options(self, base_url: str, api_key: str) -> list[ResumeOption]:
        resolved_base_url = self._resolve_rxresume_base_url(base_url)
        resolved_api_key = self._resolve_secret("RX_RESUME_API_KEY", api_key)
        if not resolved_api_key:
            raise ValueError("Provide a Reactive Resume API key during setup or in .env.")
        return self.rxresume_options_loader(resolved_base_url, resolved_api_key)

    def _load_candidate_profile(
        self,
        *,
        source_type: str,
        pdf_bytes: bytes | None,
        pdf_filename: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        openai_api_key: str,
        client: Any | None = None,
    ) -> CandidateProfile:
        source = source_type.strip().lower()
        if source == "pdf":
            if not pdf_bytes:
                raise ValueError("Upload or select a PDF resume before searching.")
            resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
            profile = self.pdf_loader(
                pdf_bytes,
                resolved_openai_key,
                filename=pdf_filename or "resume.pdf",
                client=client,
            )
            return ensure_candidate_profile_has_signal(
                profile,
                error_message="Could not extract enough resume details from the PDF.",
            )

        if source == "rxresume":
            resolved_base_url = self._resolve_rxresume_base_url(rxresume_base_url)
            resolved_api_key = self._resolve_secret("RX_RESUME_API_KEY", rxresume_api_key)
            if not resolved_api_key:
                raise ValueError("Provide a Reactive Resume API key during setup or in .env.")
            if not rxresume_resume_id.strip():
                raise ValueError("Select a Reactive Resume entry before searching.")
            profile = self.rxresume_profile_loader(
                resolved_base_url,
                resolved_api_key,
                rxresume_resume_id.strip(),
            )
            return ensure_candidate_profile_has_signal(
                profile,
                error_message="Could not extract enough resume details from Reactive Resume.",
            )

        raise ValueError("Choose a supported resume source.")

    def preview_profile(
        self,
        *,
        source_type: str,
        pdf_bytes: bytes | None,
        pdf_filename: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        openai_api_key: str,
    ) -> dict[str, Any]:
        resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
        if source_type.strip().lower() == "pdf" and not resolved_openai_key:
            raise ValueError("Provide an OpenAI API key during setup or in .env.")

        openai_client = OpenAI(api_key=resolved_openai_key) if resolved_openai_key else None
        profile = self._load_candidate_profile(
            source_type=source_type,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key=rxresume_api_key,
            rxresume_resume_id=rxresume_resume_id,
            openai_api_key=resolved_openai_key,
            client=openai_client,
        )
        return {
            "profile": profile,
            "location_used": profile.inferred_location,
            "status": "Resume analyzed.",
            "search_terms_text": _search_terms_text_for_profile(profile),
        }

    def run_search(
        self,
        *,
        source_type: str,
        pdf_bytes: bytes | None,
        pdf_filename: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        openai_api_key: str,
        serpapi_api_key: str,
        location_override: str,
        include_remote: bool,
        search_terms_text: str = "",
        candidate_profile: CandidateProfile | dict[str, Any] | None = None,
    ) -> SearchRunResult:
        source = source_type.strip().lower()
        resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
        resolved_serpapi_key = self._resolve_secret("SERPAPI_API_KEY", serpapi_api_key)
        if not resolved_openai_key:
            raise ValueError("Provide an OpenAI API key during setup or in .env.")
        if not resolved_serpapi_key:
            raise ValueError("Provide a SerpApi key during setup or in .env.")

        openai_client = OpenAI(api_key=resolved_openai_key)
        if candidate_profile is not None:
            profile = ensure_candidate_profile_has_signal(
                CandidateProfile.model_validate(candidate_profile),
                error_message="Could not extract enough resume details from the selected resume.",
            )
        else:
            profile = self._load_candidate_profile(
                source_type=source,
                pdf_bytes=pdf_bytes,
                pdf_filename=pdf_filename,
                rxresume_base_url=rxresume_base_url,
                rxresume_api_key=rxresume_api_key,
                rxresume_resume_id=rxresume_resume_id,
                openai_api_key=resolved_openai_key,
                client=openai_client,
            )

        location_used = location_override.strip() or profile.inferred_location.strip()
        request = SearchRequest(
            source_type=source,
            location_override=location_used,
            search_terms=_parse_search_terms_text(search_terms_text),
            include_remote=include_remote,
        )
        provider = self.provider_factory(resolved_serpapi_key)
        matches = self.matcher(
            profile,
            request,
            provider,
            openai_key=resolved_openai_key,
            openai_client=openai_client,
        )

        status = (
            f"Found {len(matches)} matching jobs."
            if matches
            else "No jobs met the 7/10 threshold."
        )
        return SearchRunResult(
            profile=profile,
            location_used=location_used,
            matches=matches,
            status=status,
        )


class AppController:
    def __init__(self, service: Any | None = None, workspace: LocalWorkspace | None = None) -> None:
        self.workspace = workspace or getattr(service, "workspace", None) or LocalWorkspace()
        self.service = service or JobMatchService(workspace=self.workspace)

    def saved_resume_choices(self) -> list[tuple[str, str]]:
        return [(resume.name, resume.name) for resume in self.workspace.list_saved_resumes()]

    def default_saved_resume_name(self) -> str | None:
        choices = self.saved_resume_choices()
        return choices[0][1] if choices else None

    def default_rxresume_api_url(self) -> str:
        return self.workspace.resolve_value("RX_RESUME_API_URL") or DEFAULT_RXRESUME_RESUMES_URL

    def setup_required(self) -> bool:
        return not self.workspace.env_exists()

    def current_settings(self) -> dict[str, str]:
        values = self.workspace.load_env_values()
        return {
            "openai_api_key": values.get("OPENAI_API_KEY", ""),
            "serpapi_api_key": values.get("SERPAPI_API_KEY", ""),
            "rxresume_api_key": values.get("RX_RESUME_API_KEY", ""),
            "rxresume_api_url": values.get("RX_RESUME_API_URL", "") or DEFAULT_RXRESUME_RESUMES_URL,
        }

    def complete_setup(
        self,
        *,
        openai_api_key: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
        source_type: str,
        pdf_file: str | None,
    ) -> dict[str, Any]:
        if not openai_api_key.strip():
            raise ValueError("Provide an OpenAI API key to continue.")
        if not serpapi_api_key.strip():
            raise ValueError("Provide a SerpApi key to continue.")

        saved_resume_name: str | None = None
        chosen_source = source_type.strip().lower()
        if chosen_source == "pdf":
            if not pdf_file:
                raise ValueError("Upload a PDF resume to continue.")
            saved_resume_name = self.workspace.save_uploaded_resume(pdf_file).name
        elif chosen_source == "rxresume" and not rxresume_api_key.strip():
            raise ValueError("Provide a Reactive Resume API key if you want to start with Reactive Resume.")

        self.workspace.save_env_values(
            {
                "OPENAI_API_KEY": openai_api_key,
                "SERPAPI_API_KEY": serpapi_api_key,
                "RX_RESUME_API_KEY": rxresume_api_key,
                "RX_RESUME_API_URL": rxresume_api_url or DEFAULT_RXRESUME_RESUMES_URL,
            }
        )

        return {
            "status": "Setup saved. Continue with job search.",
            "source_type": chosen_source,
            "saved_resume_name": saved_resume_name or self.default_saved_resume_name(),
            "saved_resume_choices": self.saved_resume_choices(),
            "rxresume_api_url": self.default_rxresume_api_url(),
        }

    def save_settings(
        self,
        *,
        openai_api_key: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
    ) -> dict[str, str]:
        if not openai_api_key.strip():
            raise ValueError("Provide an OpenAI API key to save settings.")
        if not serpapi_api_key.strip():
            raise ValueError("Provide a SerpApi key to save settings.")

        merged = self.workspace.save_env_values(
            {
                "OPENAI_API_KEY": openai_api_key,
                "SERPAPI_API_KEY": serpapi_api_key,
                "RX_RESUME_API_KEY": rxresume_api_key,
                "RX_RESUME_API_URL": rxresume_api_url or DEFAULT_RXRESUME_RESUMES_URL,
            }
        )
        return {
            "status": "Settings saved.",
            "openai_api_key": merged.get("OPENAI_API_KEY", ""),
            "serpapi_api_key": merged.get("SERPAPI_API_KEY", ""),
            "rxresume_api_key": merged.get("RX_RESUME_API_KEY", ""),
            "rxresume_api_url": merged.get("RX_RESUME_API_URL", DEFAULT_RXRESUME_RESUMES_URL),
        }

    def load_rxresume_options(self, base_url: str, api_key: str = "") -> tuple[str, list[tuple[str, str]]]:
        try:
            options = self.service.load_rxresume_options(base_url, api_key)
        except Exception as exc:
            return (str(exc), [])
        return (f"Loaded {len(options)} resumes.", [(option.label, option.id) for option in options])

    def preview_profile(
        self,
        *,
        source_type: str,
        pdf_bytes: bytes | None,
        pdf_filename: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        openai_api_key: str,
    ) -> dict[str, Any]:
        try:
            result = self.service.preview_profile(
                source_type=source_type,
                pdf_bytes=pdf_bytes,
                pdf_filename=pdf_filename,
                rxresume_base_url=rxresume_base_url,
                rxresume_api_key=rxresume_api_key,
                rxresume_resume_id=rxresume_resume_id,
                openai_api_key=openai_api_key,
            )
        except Exception as exc:
            return {
                "status": str(exc),
                "location_used": "",
                "profile_markdown": "",
                "candidate_profile": None,
                "search_terms_text": "",
            }

        if isinstance(result, dict):
            profile = result.get("profile")
            location_used = str(result.get("location_used") or "").strip()
            status = str(result.get("status") or "").strip()
            search_terms_text = str(result.get("search_terms_text") or "").strip()
        else:
            profile = result.profile
            location_used = result.location_used
            status = result.status
            search_terms_text = _search_terms_text_for_profile(result.profile)

        return {
            "status": status,
            "location_used": location_used,
            "profile_markdown": _profile_markdown(profile, location_used=location_used) if profile else "",
            "candidate_profile": profile.model_dump() if profile else None,
            "search_terms_text": search_terms_text,
        }

    def run_search(
        self,
        *,
        source_type: str,
        pdf_bytes: bytes | None,
        pdf_filename: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        openai_api_key: str,
        serpapi_api_key: str,
        location_override: str,
        include_remote: bool,
        search_terms_text: str = "",
        candidate_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = source_type.strip().lower()
        if source == "rxresume" and not rxresume_resume_id.strip():
            return {
                "status": "Select a Reactive Resume entry before searching.",
                "location_used": "",
                "profile_markdown": "",
                "search_terms_text": search_terms_text,
                "candidate_profile": candidate_profile,
                "rows": [],
                "results_markdown": "",
            }

        try:
            result = self.service.run_search(
                source_type=source_type,
                pdf_bytes=pdf_bytes,
                pdf_filename=pdf_filename,
                rxresume_base_url=rxresume_base_url,
                rxresume_api_key=rxresume_api_key,
                rxresume_resume_id=rxresume_resume_id,
                openai_api_key=openai_api_key,
                serpapi_api_key=serpapi_api_key,
                location_override=location_override,
                include_remote=include_remote,
                search_terms_text=search_terms_text,
                candidate_profile=candidate_profile,
            )
        except Exception as exc:
            return {
                "status": str(exc),
                "location_used": "",
                "profile_markdown": "",
                "search_terms_text": search_terms_text,
                "candidate_profile": candidate_profile,
                "rows": [],
                "results_markdown": "",
            }

        if isinstance(result, dict):
            profile = result.get("profile")
            location_used = str(result.get("location_used") or "").strip()
            matches = result.get("matches") or []
            status = str(result.get("status") or "").strip()
        else:
            profile = result.profile
            location_used = result.location_used
            matches = result.matches
            status = result.status

        rows = _rows_from_matches(matches)
        return {
            "status": status,
            "location_used": location_used,
            "profile_markdown": _profile_markdown(profile, location_used=location_used) if profile else "",
            "search_terms_text": search_terms_text or (_search_terms_text_for_profile(profile) if profile else ""),
            "candidate_profile": profile.model_dump() if profile else candidate_profile,
            "rows": rows,
            "results_markdown": _results_markdown(matches),
        }


def build_app(
    controller: AppController | None = None,
    *,
    workspace: LocalWorkspace | None = None,
) -> gr.Blocks:
    workspace = workspace or LocalWorkspace()
    controller = controller or AppController(workspace=workspace)

    settings_values = controller.current_settings()

    def saved_resume_update(value: str | None = None) -> dict[str, Any]:
        choices = controller.saved_resume_choices()
        selected = value if value and any(option[1] == value for option in choices) else None
        if selected is None and choices:
            selected = choices[0][1]
        return gr.update(choices=choices, value=selected)

    def resolve_pdf_source(saved_resume_name: str | None) -> tuple[bytes | None, str]:
        saved_resume = workspace.get_saved_resume(saved_resume_name or "")
        if saved_resume is None:
            return (None, "")
        return (saved_resume.path.read_bytes(), saved_resume.path.name)

    def current_resume_token(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
    ) -> str:
        source = source_type.strip().lower()
        if source == "pdf":
            return f"pdf:{(saved_resume_name or '').strip()}"
        if source == "rxresume":
            return f"rxresume:{(rxresume_resume_id or '').strip()}"
        return source

    def toggle_search_source(source_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
        source = source_type.strip().lower()
        return (
            gr.update(visible=source == "pdf"),
            gr.update(visible=source == "rxresume"),
        )

    def toggle_setup_source(source_type: str) -> dict[str, Any]:
        return gr.update(visible=source_type.strip().lower() == "pdf")

    def toggle_settings_ui(current_visible: bool) -> tuple[bool, dict[str, Any]]:
        next_visible = not current_visible
        return next_visible, gr.update(visible=next_visible)

    def invalidate_analysis_ui() -> tuple[str, str, str, str, None, pd.DataFrame, str, dict[str, Any], str]:
        return (
            "Analyze the selected resume to continue.",
            "",
            "",
            "",
            None,
            _empty_results_frame(),
            "",
            gr.update(interactive=False),
            "",
        )

    def store_uploaded_pdf_ui(
        uploaded_pdf: str | None,
    ) -> tuple[dict[str, Any], str, str, str, str, None, pd.DataFrame, str, dict[str, Any], str]:
        if not uploaded_pdf:
            status, location, search_terms, profile_markdown, candidate_profile, frame, results, button, token = (
                invalidate_analysis_ui()
            )
            return (
                saved_resume_update(controller.default_saved_resume_name()),
                status,
                location,
                search_terms,
                profile_markdown,
                candidate_profile,
                frame,
                results,
                button,
                token,
            )

        saved_resume = workspace.save_uploaded_resume(uploaded_pdf)
        return (
            saved_resume_update(saved_resume.name),
            f"Saved {saved_resume.name} to `.resume`. Analyze the selected resume to continue.",
            "",
            "",
            "",
            None,
            _empty_results_frame(),
            "",
            gr.update(interactive=False),
            "",
        )

    def save_setup_ui(
        openai_api_key: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
        source_type: str,
        setup_pdf_file: str | None,
    ) -> tuple[
        str,
        dict[str, Any],
        dict[str, Any],
        str,
        dict[str, Any],
        str,
        dict[str, Any],
        dict[str, Any],
        str,
        str,
        str,
        str,
    ]:
        try:
            result = controller.complete_setup(
                openai_api_key=openai_api_key,
                serpapi_api_key=serpapi_api_key,
                rxresume_api_key=rxresume_api_key,
                rxresume_api_url=rxresume_api_url,
                source_type=source_type,
                pdf_file=setup_pdf_file,
            )
        except Exception as exc:
            return (
                str(exc),
                gr.update(),
                gr.update(),
                source_type,
                saved_resume_update(controller.default_saved_resume_name()),
                controller.default_rxresume_api_url(),
                gr.update(),
                gr.update(),
                openai_api_key,
                serpapi_api_key,
                rxresume_api_key,
                rxresume_api_url or controller.default_rxresume_api_url(),
            )

        selected_source = result["source_type"]
        settings_after_save = controller.current_settings()
        return (
            result["status"],
            gr.update(visible=False),
            gr.update(visible=True),
            selected_source,
            saved_resume_update(result["saved_resume_name"]),
            result["rxresume_api_url"],
            gr.update(visible=selected_source == "pdf"),
            gr.update(visible=selected_source == "rxresume"),
            settings_after_save["openai_api_key"],
            settings_after_save["serpapi_api_key"],
            settings_after_save["rxresume_api_key"],
            settings_after_save["rxresume_api_url"],
        )

    def save_settings_ui(
        openai_api_key: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
    ) -> tuple[str, str, str, str, str, str]:
        try:
            result = controller.save_settings(
                openai_api_key=openai_api_key,
                serpapi_api_key=serpapi_api_key,
                rxresume_api_key=rxresume_api_key,
                rxresume_api_url=rxresume_api_url,
            )
        except Exception as exc:
            return (
                str(exc),
                openai_api_key,
                serpapi_api_key,
                rxresume_api_key,
                rxresume_api_url or controller.default_rxresume_api_url(),
                rxresume_api_url or controller.default_rxresume_api_url(),
            )

        return (
            result["status"],
            result["openai_api_key"],
            result["serpapi_api_key"],
            result["rxresume_api_key"],
            result["rxresume_api_url"],
            result["rxresume_api_url"],
        )

    def load_resumes_ui(
        base_url: str,
    ) -> tuple[str, dict[str, Any], str, str, str, None, pd.DataFrame, str, dict[str, Any], str]:
        status, options = controller.load_rxresume_options(base_url)
        value = options[0][1] if options else None
        return (
            status,
            gr.update(choices=options, value=value),
            "",
            "",
            "",
            None,
            _empty_results_frame(),
            "",
            gr.update(interactive=False),
            "",
        )

    def preview_resume_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_base_url: str,
        rxresume_resume_id: str,
    ) -> tuple[str, str, str, str, dict[str, Any] | None, dict[str, Any], str]:
        pdf_bytes: bytes | None = None
        pdf_filename = ""

        if source_type.strip().lower() == "pdf":
            pdf_bytes, pdf_filename = resolve_pdf_source(saved_resume_name)

        result = controller.preview_profile(
            source_type=source_type,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            openai_api_key="",
        )
        current_token = ""
        if result["candidate_profile"] is not None:
            current_token = current_resume_token(source_type, saved_resume_name, rxresume_resume_id)
        return (
            result["status"],
            result["location_used"],
            result["search_terms_text"],
            result["profile_markdown"],
            result["candidate_profile"],
            gr.update(interactive=result["candidate_profile"] is not None),
            current_token,
        )

    def find_jobs_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_base_url: str,
        rxresume_resume_id: str,
        location_override: str,
        search_terms_text: str,
        include_remote: bool,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
    ) -> tuple[str, str, str, str, dict[str, Any] | None, pd.DataFrame, str, dict[str, Any], str]:
        current_token = current_resume_token(source_type, saved_resume_name, rxresume_resume_id)
        if candidate_profile is None or not analysis_token or analysis_token != current_token:
            return (
                "Analyze the selected resume before searching.",
                "",
                search_terms_text,
                "",
                None,
                _empty_results_frame(),
                "",
                gr.update(interactive=False),
                "",
            )

        pdf_bytes: bytes | None = None
        pdf_filename = ""
        if source_type.strip().lower() == "pdf":
            pdf_bytes, pdf_filename = resolve_pdf_source(saved_resume_name)

        result = controller.run_search(
            source_type=source_type,
            pdf_bytes=pdf_bytes,
            pdf_filename=pdf_filename,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            openai_api_key="",
            serpapi_api_key="",
            location_override=location_override,
            include_remote=include_remote,
            search_terms_text=search_terms_text,
            candidate_profile=candidate_profile,
        )
        frame = pd.DataFrame(result["rows"])
        return (
            result["status"],
            result["location_used"],
            result["search_terms_text"],
            result["profile_markdown"],
            result["candidate_profile"],
            frame,
            result["results_markdown"],
            gr.update(interactive=result["candidate_profile"] is not None),
            current_token,
        )

    with gr.Blocks(title="Resume to Jobs Finder", css=APP_CSS) as demo:
        gr.Markdown(
            """
            # Resume to Jobs Finder

            Save your API keys once, analyze a resume, then tune the search terms before finding jobs.
            """
        )

        with gr.Group(visible=controller.setup_required()) as setup_group:
            gr.Markdown("## Setup")
            setup_openai_api_key = gr.Textbox(label="OpenAI API key", type="password")
            setup_serpapi_api_key = gr.Textbox(label="SerpApi key", type="password")
            setup_rxresume_api_key = gr.Textbox(label="Reactive Resume API key (Optional)", type="password")
            setup_rxresume_api_url = gr.Textbox(
                label="Reactive Resume API URL (Optional)",
                value=controller.default_rxresume_api_url(),
                placeholder=DEFAULT_RXRESUME_RESUMES_URL,
            )
            setup_source_type = gr.Radio(
                label="Initial resume source",
                choices=[("PDF", "pdf"), ("Reactive Resume", "rxresume")],
                value="pdf",
            )
            with gr.Group(visible=True) as setup_pdf_group:
                setup_pdf_file = gr.File(label="Upload PDF to store in .resume", file_types=[".pdf"], type="filepath")
            save_setup_button = gr.Button("Save setup and continue", variant="primary")
            setup_status = gr.Markdown()

        with gr.Group(visible=not controller.setup_required()) as search_group:
            with gr.Row(elem_id="search-header"):
                with gr.Column(scale=6):
                    gr.Markdown("## Job Search")
                    gr.Markdown("Keys are loaded from `.env` when available. Uploaded PDFs are stored in `.resume`.")
                with gr.Column(scale=1, min_width=120):
                    settings_toggle_button = gr.Button("Settings", size="sm", elem_id="search-settings-toggle")
            settings_panel_visible = gr.State(value=False)
            with gr.Group(visible=False, elem_id="settings-panel") as settings_group:
                settings_openai_api_key = gr.Textbox(
                    label="OpenAI API key",
                    type="password",
                    value=settings_values["openai_api_key"],
                )
                settings_serpapi_api_key = gr.Textbox(
                    label="SerpApi key",
                    type="password",
                    value=settings_values["serpapi_api_key"],
                )
                settings_rxresume_api_key = gr.Textbox(
                    label="Reactive Resume API key (Optional)",
                    type="password",
                    value=settings_values["rxresume_api_key"],
                )
                settings_rxresume_api_url = gr.Textbox(
                    label="Reactive Resume API URL",
                    value=settings_values["rxresume_api_url"],
                    placeholder=DEFAULT_RXRESUME_RESUMES_URL,
                )
                save_settings_button = gr.Button("Save settings", variant="primary")
                settings_status = gr.Markdown()
            source_type = gr.Radio(
                label="Resume source",
                choices=[("PDF", "pdf"), ("Reactive Resume", "rxresume")],
                value="pdf",
            )

            with gr.Group(visible=True) as pdf_group:
                saved_pdf_name = gr.Dropdown(
                    label="Saved PDF resumes",
                    choices=controller.saved_resume_choices(),
                    value=controller.default_saved_resume_name(),
                )
                uploaded_pdf = gr.File(label="Upload a new PDF resume", file_types=[".pdf"], type="filepath")
                browse_pdf_button = gr.UploadButton(
                    "Browse PDF",
                    file_types=[".pdf"],
                    file_count="single",
                    type="filepath",
                    size="sm",
                )

            with gr.Group(visible=False) as rxresume_group:
                rxresume_base_url = gr.Textbox(
                    label="Reactive Resume API URL",
                    value=controller.default_rxresume_api_url(),
                    placeholder=DEFAULT_RXRESUME_RESUMES_URL,
                )
                load_resumes_button = gr.Button("Load resumes")
                rxresume_resume_id = gr.Dropdown(label="Reactive Resume entry", choices=[])

            analyze_resume_button = gr.Button("Analyze resume")
            with gr.Row():
                location_override = gr.Textbox(label="Location", placeholder="Leave blank to use resume location")
                include_remote = gr.Checkbox(label="Include remote jobs", value=True)

            search_terms_text = gr.Textbox(
                label="Search terms",
                lines=3,
                placeholder="One search term per line, for example:\nMachine Learning Engineer\nApplied Scientist",
            )
            find_jobs_button = gr.Button("Find jobs", variant="primary", interactive=False)
            status = gr.Markdown()
            profile_markdown = gr.Markdown()
            candidate_profile_state = gr.State(value=None)
            analysis_token_state = gr.State(value="")
            results_table = gr.Dataframe(
                value=_empty_results_frame(),
                headers=["score", "title", "company", "location", "pay_range", "via", "apply_url", "rationale"],
                interactive=False,
                wrap=False,
                row_count=10,
                max_height=720,
                show_fullscreen_button=True,
                column_widths=[80, 300, 220, 220, 180, 180, 320, 420],
                elem_id="job-results-table",
            )
            results_markdown = gr.Markdown()

        setup_source_type.change(
            toggle_setup_source,
            inputs=[setup_source_type],
            outputs=[setup_pdf_group],
        )
        save_setup_button.click(
            save_setup_ui,
            inputs=[
                setup_openai_api_key,
                setup_serpapi_api_key,
                setup_rxresume_api_key,
                setup_rxresume_api_url,
                setup_source_type,
                setup_pdf_file,
            ],
            outputs=[
                setup_status,
                setup_group,
                search_group,
                source_type,
                saved_pdf_name,
                rxresume_base_url,
                pdf_group,
                rxresume_group,
                settings_openai_api_key,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
            ],
            queue=False,
        )
        settings_toggle_button.click(
            toggle_settings_ui,
            inputs=[settings_panel_visible],
            outputs=[settings_panel_visible, settings_group],
            queue=False,
        )
        save_settings_button.click(
            save_settings_ui,
            inputs=[
                settings_openai_api_key,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
            ],
            outputs=[
                settings_status,
                settings_openai_api_key,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
                rxresume_base_url,
            ],
            queue=False,
        )
        source_type.change(
            toggle_search_source,
            inputs=[source_type],
            outputs=[pdf_group, rxresume_group],
        )
        source_type.change(
            invalidate_analysis_ui,
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        uploaded_pdf.change(
            store_uploaded_pdf_ui,
            inputs=[uploaded_pdf],
            outputs=[
                saved_pdf_name,
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        browse_pdf_button.upload(
            lambda pdf_path: pdf_path,
            inputs=[browse_pdf_button],
            outputs=[uploaded_pdf],
            queue=False,
        )
        saved_pdf_name.change(
            invalidate_analysis_ui,
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        load_resumes_button.click(
            load_resumes_ui,
            inputs=[rxresume_base_url],
            outputs=[
                status,
                rxresume_resume_id,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        rxresume_base_url.change(
            invalidate_analysis_ui,
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        rxresume_resume_id.change(
            invalidate_analysis_ui,
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        analyze_resume_button.click(
            preview_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_base_url,
                rxresume_resume_id,
            ],
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )
        find_jobs_button.click(
            find_jobs_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_base_url,
                rxresume_resume_id,
                location_override,
                search_terms_text,
                include_remote,
                candidate_profile_state,
                analysis_token_state,
            ],
            outputs=[
                status,
                location_override,
                search_terms_text,
                profile_markdown,
                candidate_profile_state,
                results_table,
                results_markdown,
                find_jobs_button,
                analysis_token_state,
            ],
            queue=False,
        )

    return demo
