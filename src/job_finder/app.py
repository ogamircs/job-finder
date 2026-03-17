from __future__ import annotations

import html
from typing import Any

import gradio as gr
import pandas as pd
from openai import OpenAI

from .application_documents import ApplicationArtifactsService
from .job_provider import SerpApiGoogleJobsProvider
from .matching import build_search_queries, find_job_matches
from .models import CandidateProfile, ResumeOption, SavedJobRecord, ScoredJobMatch, SearchRequest, SearchRunResult
from .resume_sources import (
    DEFAULT_RXRESUME_RESUMES_URL,
    ensure_candidate_profile_has_signal,
    list_rxresume_resumes,
    load_candidate_profile_from_pdf,
    load_candidate_profile_from_rxresume,
)
from .saved_jobs import SavedJobsStore, saved_job_identity
from .workspace import LocalWorkspace

DEFAULT_OPENAI_MODEL = "gpt-4o"
RESULTS_TABLE_HEADERS = ["Score", "Title", "Company", "Location", "Pay Range", "Source", "Apply", "Saved"]
SAVED_JOBS_TABLE_HEADERS = ["Score", "Title", "Company", "Location", "Pay Range", "Source", "Apply", "Updated"]
RESULT_SORT_CHOICES = ["Best match", "Company", "Location", "Newest"]

APP_CSS = """
#app-header {
    align-items: center;
    gap: 16px;
}

#top-actions {
    justify-content: flex-end;
    align-items: center;
    gap: 12px;
}

#settings-panel,
.step-card,
.detail-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 18px;
    background: var(--block-background-fill);
}

#settings-panel {
    margin-bottom: 18px;
}

.setup-copy,
.workspace-copy {
    color: var(--body-text-color-subdued);
    margin-top: 4px;
}

#job-results-table,
#saved-jobs-table {
    overflow-x: auto;
}

#job-results-table .table-wrap,
#job-results-table .table-container,
#job-results-table .wrap,
#job-results-table > .wrap,
#saved-jobs-table .table-wrap,
#saved-jobs-table .table-container,
#saved-jobs-table .wrap,
#saved-jobs-table > .wrap {
    overflow-x: auto !important;
}

#job-results-table table,
#saved-jobs-table table {
    min-width: 980px;
}

.step-card {
    margin-bottom: 14px;
}

.step-card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
}

.step-card-title h3 {
    margin: 0;
    font-size: 1.05rem;
}

.step-pill {
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}

.step-pill.current {
    background: rgba(59, 130, 246, 0.14);
    color: #2563eb;
}

.step-pill.ready {
    background: rgba(245, 158, 11, 0.14);
    color: #b45309;
}

.step-pill.complete {
    background: rgba(22, 163, 74, 0.14);
    color: #15803d;
}

.status-banner {
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 0.95rem;
}

.status-banner.info {
    background: rgba(59, 130, 246, 0.1);
    color: #1d4ed8;
}

.status-banner.success {
    background: rgba(22, 163, 74, 0.12);
    color: #166534;
}

.status-banner.warning {
    background: rgba(245, 158, 11, 0.12);
    color: #92400e;
}

.status-banner.error {
    background: rgba(239, 68, 68, 0.12);
    color: #b91c1c;
}

.summary-card h4,
.job-detail-card h4,
.saved-job-summary h4 {
    margin: 0 0 10px 0;
}

.summary-grid,
.job-meta-grid {
    display: grid;
    gap: 10px 14px;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    margin-bottom: 12px;
}

.summary-label {
    color: var(--body-text-color-subdued);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.summary-value {
    font-weight: 600;
}

.summary-list {
    margin: 0;
    padding-left: 18px;
}

.detail-body {
    white-space: pre-wrap;
    line-height: 1.5;
}

.saved-count-chip {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 4px 10px;
    background: rgba(15, 23, 42, 0.08);
    font-size: 0.82rem;
    font-weight: 600;
}

.tab-copy {
    color: var(--body-text-color-subdued);
    margin: 0 0 14px 0;
}
"""


def _artifact_status_lookup(artifacts_state: dict[str, Any] | None) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for key, artifact in (artifacts_state or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(artifact, dict):
            lookup[index] = str(artifact.get("status") or "Ready").strip() or "Ready"
    return lookup


def _saved_jobs_tab_label(count: int) -> str:
    return f"Saved Jobs ({max(int(count), 0)})"


def _saved_jobs_button_label(count: int) -> str:
    return _saved_jobs_tab_label(count)


def _rows_from_matches(
    matches: list[Any],
    *,
    linkify_apply_url: bool = False,
    saved_jobs_state: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    saved_identities = {
        saved_job_identity(record.match)
        for record in (
            SavedJobRecord.model_validate(item)
            for item in (saved_jobs_state or [])
        )
    }
    for index, match in enumerate(matches):
        normalized_match = ScoredJobMatch.model_validate(match)
        rows.append(
            {
                "Score": normalized_match.score_10,
                "Title": normalized_match.job.title,
                "Company": normalized_match.job.company,
                "Location": normalized_match.job.location or ("Remote" if normalized_match.job.remote_flag else ""),
                "Pay Range": normalized_match.job.pay_range,
                "Source": normalized_match.job.via,
                "Apply": (
                    _format_link(normalized_match.job.apply_url)
                    if linkify_apply_url
                    else normalized_match.job.apply_url
                ),
                "Saved": "Saved" if saved_job_identity(normalized_match) in saved_identities else "",
            }
        )
    return rows


def _rows_from_saved_jobs(saved_jobs: list[Any], *, linkify_apply_url: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in saved_jobs:
        rows.append(
            {
                "Score": record.match.score_10,
                "Title": record.match.job.title,
                "Company": record.match.job.company,
                "Location": record.match.job.location or ("Remote" if record.match.job.remote_flag else ""),
                "Pay Range": record.match.job.pay_range,
                "Source": record.match.job.via,
                "Apply": (
                    _format_link(record.match.job.apply_url)
                    if linkify_apply_url
                    else record.match.job.apply_url
                ),
                "Updated": record.updated_at,
            }
        )
    return rows


def _format_link(url: str, label: str | None = None) -> str:
    cleaned = str(url or "").strip()
    if not cleaned:
        return ""
    escaped_url = html.escape(cleaned, quote=True)
    escaped_label = html.escape(label or cleaned)
    return f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">{escaped_label}</a>'


def _status_tone(message: str) -> str:
    lowered = message.strip().casefold()
    if not lowered:
        return "info"
    if any(token in lowered for token in ("error", "failed", "invalid", "400", "401", "403", "404")):
        return "error"
    if any(token in lowered for token in ("select", "provide", "upload", "no jobs", "before", "choose")):
        return "warning"
    if any(token in lowered for token in ("saved", "loaded", "found", "generated", "updated", "analyzed", "deleted")):
        return "success"
    return "info"


def _status_html(message: str) -> str:
    cleaned = str(message or "").strip()
    if not cleaned:
        return ""
    tone = _status_tone(cleaned)
    return f'<div class="status-banner {tone}">{html.escape(cleaned)}</div>'


def _step_header_html(number: int, title: str, state: str) -> str:
    normalized = state.strip().lower() or "ready"
    return (
        '<div class="step-card-title">'
        f"<h3>{number}. {html.escape(title)}</h3>"
        f'<span class="step-pill {html.escape(normalized)}">{html.escape(state)}</span>'
        "</div>"
    )


def _list_badges(values: list[str]) -> str:
    if not values:
        return "<span class=\"summary-value\">None yet</span>"
    items = "".join(f"<li>{html.escape(value)}</li>" for value in values)
    return f'<ul class="summary-list">{items}</ul>'


def _candidate_summary_html(profile: CandidateProfile | dict[str, Any] | None, *, location_used: str) -> str:
    if profile is None:
        return '<div class="summary-card"><h4>No analysis yet</h4><p>Analyze the selected resume to generate a candidate summary and suggested search terms.</p></div>'
    candidate = CandidateProfile.model_validate(profile)
    return (
        '<div class="summary-card">'
        f"<h4>{html.escape(candidate.name or 'Candidate profile')}</h4>"
        '<div class="summary-grid">'
        f'<div><div class="summary-label">Headline</div><div class="summary-value">{html.escape(candidate.headline or "Not provided")}</div></div>'
        f'<div><div class="summary-label">Location</div><div class="summary-value">{html.escape(location_used or "Remote only")}</div></div>'
        f'<div><div class="summary-label">Experience</div><div class="summary-value">{html.escape(f"{candidate.years_experience:g} years" if candidate.years_experience else "Not inferred")}</div></div>'
        "</div>"
        '<div class="summary-grid">'
        f'<div><div class="summary-label">Target titles</div>{_list_badges(candidate.target_titles)}</div>'
        f'<div><div class="summary-label">Top skills</div>{_list_badges(candidate.top_skills[:8])}</div>'
        "</div>"
        "</div>"
    )


def _resume_source_summary_html(
    *,
    source_type: str,
    saved_resume_name: str | None,
    rxresume_resume_id: str,
    rxresume_options: list[tuple[str, str]] | None = None,
) -> str:
    source = source_type.strip().lower()
    if source == "pdf":
        selected = saved_resume_name or "No PDF selected"
        return (
            '<div class="summary-card">'
            "<h4>Selected resume source</h4>"
            '<div class="summary-grid">'
            '<div><div class="summary-label">Source</div><div class="summary-value">PDF</div></div>'
            f'<div><div class="summary-label">Resume</div><div class="summary-value">{html.escape(selected)}</div></div>'
            "</div></div>"
        )

    label_lookup = {value: label for label, value in (rxresume_options or [])}
    selected_label = label_lookup.get(rxresume_resume_id, rxresume_resume_id or "No Reactive Resume entry selected")
    return (
        '<div class="summary-card">'
        "<h4>Selected resume source</h4>"
        '<div class="summary-grid">'
        '<div><div class="summary-label">Source</div><div class="summary-value">Reactive Resume</div></div>'
        f'<div><div class="summary-label">Resume</div><div class="summary-value">{html.escape(selected_label)}</div></div>'
        "</div></div>"
    )


def _job_detail_html(match: ScoredJobMatch | dict[str, Any]) -> str:
    normalized = ScoredJobMatch.model_validate(match)
    location_text = normalized.job.location or ("Remote" if normalized.job.remote_flag else "Unspecified")
    links = [_format_link(normalized.job.apply_url)]
    if normalized.job.share_url and normalized.job.share_url != normalized.job.apply_url:
        links.append(_format_link(normalized.job.share_url, "Share link"))
    return (
        '<div class="job-detail-card">'
        f"<h4>{html.escape(normalized.job.title)}</h4>"
        '<div class="job-meta-grid">'
        f'<div><div class="summary-label">Company</div><div class="summary-value">{html.escape(normalized.job.company)}</div></div>'
        f'<div><div class="summary-label">Location</div><div class="summary-value">{html.escape(location_text)}</div></div>'
        f'<div><div class="summary-label">Score</div><div class="summary-value">{html.escape(str(normalized.score_10))}/10</div></div>'
        f'<div><div class="summary-label">Pay Range</div><div class="summary-value">{html.escape(normalized.job.pay_range or "Not listed")}</div></div>'
        "</div>"
        f'<p><strong>Why it matches:</strong> {html.escape(normalized.rationale or "No rationale available.")}</p>'
        f'<p><strong>Matched skills:</strong> {html.escape(", ".join(normalized.matched_skills) or "None noted")}</p>'
        f'<p><strong>Missing signals:</strong> {html.escape(", ".join(normalized.missing_signals) or "None noted")}</p>'
        f'<p class="detail-body">{html.escape(normalized.job.description or "No description available.")}</p>'
        f'<p>{" | ".join(links)}</p>'
        "</div>"
    )


def _saved_job_summary_html(record: SavedJobRecord | dict[str, Any]) -> str:
    normalized = SavedJobRecord.model_validate(record)
    return _job_detail_html(normalized.match) + (
        '<div class="summary-card" style="margin-top:12px;">'
        '<div class="summary-grid">'
        f'<div><div class="summary-label">Saved</div><div class="summary-value">{html.escape(normalized.updated_at or normalized.created_at)}</div></div>'
        f'<div><div class="summary-label">Provider</div><div class="summary-value">{html.escape(normalized.match.job.provider)}</div></div>'
        "</div></div>"
    )


def _results_markdown(matches: list[Any]) -> str:
    if not matches:
        return "No jobs met the 7/10 threshold."
    lines: list[str] = []
    for raw_match in matches:
        match = ScoredJobMatch.model_validate(raw_match)
        location_text = match.job.location or ("Remote" if match.job.remote_flag else "Unspecified")
        lines.append(
            f"- **{match.job.title}** at **{match.job.company}** in {location_text} ({match.score_10}/10) - [Apply]({match.job.apply_url})"
        )
    return "\n".join(lines)


def _selected_job_markdown(match: ScoredJobMatch | dict[str, Any]) -> str:
    normalized = ScoredJobMatch.model_validate(match)
    location_text = normalized.job.location or ("Remote" if normalized.job.remote_flag else "Unspecified")
    return (
        f"**Selected job:** {normalized.job.title} at {normalized.job.company} in {location_text}\n\n"
        f"[Apply]({normalized.job.apply_url})"
    )


def _apply_link_button_html(url: str) -> str:
    if not str(url or "").strip():
        return ""
    return (
        '<div style="display:flex; align-items:center; min-height:40px;">'
        f'{_format_link(url, "Apply")}'
        "</div>"
    )


def _profile_markdown(profile: CandidateProfile, *, location_used: str) -> str:
    return _candidate_summary_html(profile, location_used=location_used)


def _posted_at_rank(posted_at: str) -> int:
    cleaned = str(posted_at or "").strip().casefold()
    if not cleaned:
        return 10_000
    if cleaned in {"today", "just posted", "new"}:
        return 0
    if "hour" in cleaned:
        digits = "".join(character for character in cleaned if character.isdigit())
        return int(digits or "1")
    if "day" in cleaned:
        digits = "".join(character for character in cleaned if character.isdigit())
        return 24 + int(digits or "1")
    if "week" in cleaned:
        digits = "".join(character for character in cleaned if character.isdigit())
        return 24 * 7 + int(digits or "1")
    if "month" in cleaned:
        digits = "".join(character for character in cleaned if character.isdigit())
        return 24 * 30 + int(digits or "1")
    return 5_000


def _filter_and_sort_matches(
    matches: list[ScoredJobMatch],
    *,
    sort_by: str,
    filter_text: str,
) -> list[ScoredJobMatch]:
    filtered = matches
    token = filter_text.strip().casefold()
    if token:
        filtered = [
            match
            for match in filtered
            if token in match.job.title.casefold() or token in match.job.company.casefold()
        ]

    if sort_by == "Company":
        return sorted(filtered, key=lambda match: (match.job.company.casefold(), match.job.title.casefold()))
    if sort_by == "Location":
        return sorted(
            filtered,
            key=lambda match: (
                (match.job.location or ("Remote" if match.job.remote_flag else "")).casefold(),
                match.job.company.casefold(),
            ),
        )
    if sort_by == "Newest":
        return sorted(filtered, key=lambda match: (_posted_at_rank(match.job.posted_at), -match.score_10))
    return sorted(filtered, key=lambda match: (-match.score_10, match.job.company.casefold(), match.job.title.casefold()))


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
    return pd.DataFrame(columns=RESULTS_TABLE_HEADERS)


def _empty_saved_jobs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SAVED_JOBS_TABLE_HEADERS)


def _list_text(values: list[str]) -> str:
    return "\n".join(values)


def _parse_list_text(raw_text: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for line in raw_text.splitlines():
        for chunk in line.split(","):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            values.append(cleaned)
    return values


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
        application_artifacts_service: ApplicationArtifactsService | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader
        self.rxresume_options_loader = rxresume_options_loader
        self.rxresume_profile_loader = rxresume_profile_loader
        self.provider_factory = provider_factory or (lambda api_key: SerpApiGoogleJobsProvider(api_key=api_key))
        self.matcher = matcher
        self.workspace = workspace or LocalWorkspace()
        self.application_artifacts_service = application_artifacts_service or ApplicationArtifactsService()

    def _resolve_secret(self, env_var: str, value: str = "") -> str:
        return self.workspace.resolve_value(env_var, value)

    def _resolve_rxresume_base_url(self, value: str = "") -> str:
        return self.workspace.resolve_value("RX_RESUME_API_URL", value) or DEFAULT_RXRESUME_RESUMES_URL

    def _resolve_openai_model(self, value: str = "") -> str:
        return self.workspace.resolve_value("OPENAI_MODEL", value) or DEFAULT_OPENAI_MODEL

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
        openai_model: str = "",
        client: Any | None = None,
    ) -> CandidateProfile:
        source = source_type.strip().lower()
        if source == "pdf":
            if not pdf_bytes:
                raise ValueError("Upload or select a PDF resume before searching.")
            resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
            resolved_openai_model = self._resolve_openai_model(openai_model)
            profile = self.pdf_loader(
                pdf_bytes,
                resolved_openai_key,
                filename=pdf_filename or "resume.pdf",
                model=resolved_openai_model,
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
        openai_model: str = "",
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
            openai_model=openai_model,
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
        openai_model: str = "",
    ) -> SearchRunResult:
        source = source_type.strip().lower()
        resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
        resolved_serpapi_key = self._resolve_secret("SERPAPI_API_KEY", serpapi_api_key)
        resolved_openai_model = self._resolve_openai_model(openai_model)
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
                openai_model=resolved_openai_model,
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
            model=resolved_openai_model,
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

    def generate_custom_resume(
        self,
        *,
        source_type: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        candidate_profile: CandidateProfile | dict[str, Any],
        match: ScoredJobMatch | dict[str, Any],
        openai_api_key: str,
        openai_model: str = "",
    ) -> dict[str, str]:
        source = source_type.strip().lower()
        if source != "rxresume":
            raise ValueError("Custom resume generation is only available for Reactive Resume sources.")

        resolved_openai_key = self._resolve_secret("OPENAI_API_KEY", openai_api_key)
        resolved_openai_model = self._resolve_openai_model(openai_model)
        resolved_base_url = self._resolve_rxresume_base_url(rxresume_base_url)
        resolved_api_key = self._resolve_secret("RX_RESUME_API_KEY", rxresume_api_key)
        if not resolved_openai_key:
            raise ValueError("Provide an OpenAI API key during setup or in .env.")
        if not resolved_api_key:
            raise ValueError("Provide a Reactive Resume API key during setup or in .env.")
        if not rxresume_resume_id.strip():
            raise ValueError("Select a Reactive Resume entry before generating a custom resume.")

        artifacts = self.application_artifacts_service.generate_application_artifacts(
            rxresume_base_url=resolved_base_url,
            rxresume_api_key=resolved_api_key,
            base_resume_id=rxresume_resume_id.strip(),
            scored_job=ScoredJobMatch.model_validate(match),
            openai_api_key=resolved_openai_key,
            openai_model=resolved_openai_model,
            candidate_profile=candidate_profile,
        )
        return {
            "status": f"Custom resume and cover letter generated for {artifacts.company}.",
            "resume_pdf_path": artifacts.pdf_path,
            "cover_letter_path": artifacts.cover_letter_path,
        }


class AppController:
    def __init__(
        self,
        service: Any | None = None,
        workspace: LocalWorkspace | None = None,
        saved_jobs_store: SavedJobsStore | None = None,
    ) -> None:
        self.workspace = workspace or getattr(service, "workspace", None) or LocalWorkspace()
        self.service = service or JobMatchService(workspace=self.workspace)
        self.saved_jobs_store = saved_jobs_store or SavedJobsStore(self.workspace.saved_jobs_db_path)

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
            "openai_model": values.get("OPENAI_MODEL", "") or DEFAULT_OPENAI_MODEL,
            "serpapi_api_key": values.get("SERPAPI_API_KEY", ""),
            "rxresume_api_key": values.get("RX_RESUME_API_KEY", ""),
            "rxresume_api_url": values.get("RX_RESUME_API_URL", "") or DEFAULT_RXRESUME_RESUMES_URL,
        }

    def _saved_jobs_payload(
        self,
        *,
        status: str | None = None,
        selected_saved_job_id: int | None = None,
    ) -> dict[str, Any]:
        records = self.saved_jobs_store.list_jobs()
        count = len(records)
        return {
            "rows": _rows_from_saved_jobs(records),
            "saved_jobs_state": [record.model_dump() for record in records],
            "count": count,
            "saved_jobs_tab_label": _saved_jobs_tab_label(count),
            "saved_jobs_button_label": _saved_jobs_tab_label(count),
            "selected_saved_job_id": selected_saved_job_id,
            "status": status or (f"Loaded {count} saved jobs." if records else "No saved jobs yet."),
        }

    def list_saved_jobs(self) -> dict[str, Any]:
        return self._saved_jobs_payload()

    def save_saved_job(self, match: dict[str, Any] | None) -> dict[str, Any]:
        if match is None:
            payload = self._saved_jobs_payload(status="Select a job result before saving.")
            payload["saved_job"] = None
            payload["saved_job_created"] = False
            return payload

        save_result = self.saved_jobs_store.save_match(ScoredJobMatch.model_validate(match))
        record = save_result.record
        payload = self._saved_jobs_payload(
            status=(
                f"Saved {record.match.job.title} at {record.match.job.company}."
                if save_result.created
                else f"Already saved {record.match.job.title} at {record.match.job.company}."
            ),
            selected_saved_job_id=record.id,
        )
        payload["saved_job"] = record.model_dump()
        payload["saved_job_created"] = save_result.created
        return payload

    def update_saved_job(
        self,
        *,
        saved_job_id: int | None,
        provider: str,
        provider_job_id: str,
        title: str,
        company: str,
        location: str,
        pay_range: str,
        via: str,
        description: str,
        posted_at: str,
        remote_flag: bool,
        apply_url: str,
        share_url: str,
        score_10: float,
        rationale: str,
        matched_skills_text: str,
        missing_signals_text: str,
    ) -> dict[str, Any]:
        if saved_job_id is None:
            payload = self._saved_jobs_payload(status="Select a saved job before updating.")
            payload["saved_job"] = None
            return payload

        existing = self.saved_jobs_store.get_job(saved_job_id)
        if existing is None:
            payload = self._saved_jobs_payload(status="The selected saved job no longer exists.")
            payload["saved_job"] = None
            return payload

        try:
            updated = self.saved_jobs_store.update_job(
                saved_job_id,
                ScoredJobMatch(
                    job={
                        "provider": provider or existing.match.job.provider,
                        "provider_job_id": provider_job_id,
                        "title": title,
                        "company": company,
                        "location": location,
                        "pay_range": pay_range,
                        "via": via,
                        "description": description,
                        "posted_at": posted_at,
                        "remote_flag": remote_flag,
                        "apply_url": apply_url,
                        "share_url": share_url,
                    },
                    score_10=round(float(score_10)),
                    rationale=rationale,
                    matched_skills=_parse_list_text(matched_skills_text),
                    missing_signals=_parse_list_text(missing_signals_text),
                ),
            )
        except Exception as exc:
            payload = self._saved_jobs_payload(
                status=str(exc),
                selected_saved_job_id=existing.id,
            )
            payload["saved_job"] = existing.model_dump()
            return payload

        payload = self._saved_jobs_payload(
            status=(
                f"Updated saved job for {updated.match.job.company}."
                if updated is not None
                else "The selected saved job could not be updated."
            ),
            selected_saved_job_id=updated.id if updated is not None else None,
        )
        payload["saved_job"] = updated.model_dump() if updated is not None else None
        return payload

    def delete_saved_job(self, saved_job_id: int | None) -> dict[str, Any]:
        if saved_job_id is None:
            payload = self._saved_jobs_payload(status="Select a saved job before deleting.")
            return payload

        deleted = self.saved_jobs_store.delete_job(saved_job_id)
        payload = self._saved_jobs_payload(
            status=(
                f"Deleted saved job #{saved_job_id}."
                if deleted
                else "The selected saved job no longer exists."
            ),
        )
        payload["deleted"] = deleted
        return payload

    def complete_setup(
        self,
        *,
        openai_api_key: str,
        openai_model: str,
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
                "OPENAI_MODEL": openai_model or DEFAULT_OPENAI_MODEL,
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
        openai_model: str,
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
                "OPENAI_MODEL": openai_model or DEFAULT_OPENAI_MODEL,
                "SERPAPI_API_KEY": serpapi_api_key,
                "RX_RESUME_API_KEY": rxresume_api_key,
                "RX_RESUME_API_URL": rxresume_api_url or DEFAULT_RXRESUME_RESUMES_URL,
            }
        )
        return {
            "status": "Settings saved.",
            "openai_api_key": merged.get("OPENAI_API_KEY", ""),
            "openai_model": merged.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
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
        openai_model: str = "",
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
                openai_model=openai_model,
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
        openai_model: str = "",
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
                openai_model=openai_model,
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
            "matches_state": [
                ScoredJobMatch.model_validate(match).model_dump() if isinstance(match, dict) else match.model_dump()
                for match in matches
            ],
            "results_markdown": _results_markdown(matches),
        }

    def generate_custom_resume(
        self,
        *,
        source_type: str,
        rxresume_base_url: str,
        rxresume_api_key: str,
        rxresume_resume_id: str,
        candidate_profile: dict[str, Any] | None,
        match: dict[str, Any] | None,
        openai_api_key: str,
        openai_model: str = "",
    ) -> dict[str, str]:
        if candidate_profile is None or match is None:
            return {"status": "Select a job result before generating a custom resume."}

        try:
            return self.service.generate_custom_resume(
                source_type=source_type,
                rxresume_base_url=rxresume_base_url,
                rxresume_api_key=rxresume_api_key,
                rxresume_resume_id=rxresume_resume_id,
                candidate_profile=candidate_profile,
                match=match,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
        except Exception as exc:
            return {"status": str(exc)}


def _legacy_build_app(
    controller: AppController | None = None,
    *,
    workspace: LocalWorkspace | None = None,
) -> gr.Blocks:
    workspace = workspace or LocalWorkspace()
    controller = controller or AppController(workspace=workspace)

    settings_values = controller.current_settings()
    initial_saved_jobs = controller.list_saved_jobs()

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

    def matches_from_state(matches_state: list[dict[str, Any]] | None) -> list[ScoredJobMatch]:
        return [ScoredJobMatch.model_validate(match) for match in (matches_state or [])]

    def results_frame_from_state(
        matches_state: list[dict[str, Any]] | None,
        artifacts_state: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        rows = _rows_from_matches(
            matches_from_state(matches_state),
            artifacts_state=artifacts_state,
            linkify_apply_url=True,
        )
        return pd.DataFrame(rows) if rows else _empty_results_frame()

    def saved_jobs_from_state(saved_jobs_state: list[dict[str, Any]] | None) -> list[SavedJobRecord]:
        return [SavedJobRecord.model_validate(saved_job) for saved_job in (saved_jobs_state or [])]

    def saved_jobs_frame_from_state(saved_jobs_state: list[dict[str, Any]] | None) -> pd.DataFrame:
        rows = _rows_from_saved_jobs(
            saved_jobs_from_state(saved_jobs_state),
            linkify_apply_url=True,
        )
        return pd.DataFrame(rows) if rows else _empty_saved_jobs_frame()

    def saved_job_count_from_state(saved_jobs_state: list[dict[str, Any]] | None) -> int:
        return len(saved_jobs_state or [])

    def nav_button_updates(
        current_view: str,
        saved_jobs_state: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            gr.update(
                value="Job Search",
                variant="primary" if current_view == "search" else "secondary",
            ),
            gr.update(
                value=_saved_jobs_button_label(saved_job_count_from_state(saved_jobs_state)),
                variant="primary" if current_view == "saved" else "secondary",
            ),
        )

    def saved_job_record_by_id(
        saved_jobs_state: list[dict[str, Any]] | None,
        saved_job_id: int | None,
    ) -> SavedJobRecord | None:
        if saved_job_id is None:
            return None
        for record in saved_jobs_from_state(saved_jobs_state):
            if record.id == saved_job_id:
                return record
        return None

    def empty_saved_job_editor_ui(
        status_text: str = "Select a saved job to review or edit.",
    ) -> tuple[Any, ...]:
        return (
            None,
            None,
            "",
            status_text,
            "",
            0,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            False,
            "",
            "",
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    def saved_job_editor_for_id(
        saved_jobs_state: list[dict[str, Any]] | None,
        saved_job_id: int | None,
        *,
        default_status: str | None = None,
    ) -> tuple[Any, ...]:
        record = saved_job_record_by_id(saved_jobs_state, saved_job_id)
        if record is None:
            return empty_saved_job_editor_ui(default_status or "Select a saved job to review or edit.")
        return saved_job_editor_ui(
            record,
            status_text=default_status or f"Editing saved job #{record.id}.",
        )

    def saved_job_editor_ui(
        record: SavedJobRecord | None,
        *,
        status_text: str,
    ) -> tuple[Any, ...]:
        if record is None:
            return empty_saved_job_editor_ui(status_text)

        match = record.match
        return (
            record.id,
            match.model_dump(),
            _selected_job_markdown(match),
            status_text,
            match.job.provider,
            float(match.score_10),
            match.job.provider_job_id,
            match.job.title,
            match.job.company,
            match.job.location,
            match.job.pay_range,
            match.job.via,
            match.job.apply_url,
            match.job.share_url,
            match.job.remote_flag,
            match.job.posted_at,
            _list_text(match.matched_skills),
            _list_text(match.missing_signals),
            match.job.description,
            match.rationale,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            f"Ready to generate a custom resume for {match.job.company}.",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

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

    def open_search_view_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        search_button, saved_button = nav_button_updates("search", saved_jobs_state)
        return (
            "search",
            gr.update(visible=True),
            gr.update(visible=False),
            search_button,
            saved_button,
        )

    def open_saved_jobs_view_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
    ) -> tuple[Any, ...]:
        search_button, saved_button = nav_button_updates("saved", saved_jobs_state)
        return (
            "saved",
            gr.update(visible=False),
            gr.update(visible=True),
            search_button,
            saved_button,
            saved_jobs_frame_from_state(saved_jobs_state),
            *saved_job_editor_for_id(saved_jobs_state, selected_saved_job_id),
        )

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
        openai_model: str,
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
        str,
    ]:
        try:
            result = controller.complete_setup(
                openai_api_key=openai_api_key,
                openai_model=openai_model,
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
                openai_model or DEFAULT_OPENAI_MODEL,
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
            settings_after_save["openai_model"],
            settings_after_save["serpapi_api_key"],
            settings_after_save["rxresume_api_key"],
            settings_after_save["rxresume_api_url"],
        )

    def save_settings_ui(
        openai_api_key: str,
        openai_model: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
    ) -> tuple[str, str, str, str, str, str, str]:
        try:
            result = controller.save_settings(
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                serpapi_api_key=serpapi_api_key,
                rxresume_api_key=rxresume_api_key,
                rxresume_api_url=rxresume_api_url,
            )
        except Exception as exc:
            return (
                str(exc),
                openai_api_key,
                openai_model or DEFAULT_OPENAI_MODEL,
                serpapi_api_key,
                rxresume_api_key,
                rxresume_api_url or controller.default_rxresume_api_url(),
                rxresume_api_url or controller.default_rxresume_api_url(),
            )

        return (
            result["status"],
            result["openai_api_key"],
            result["openai_model"],
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

    def reset_custom_resume_ui() -> tuple[
        None,
        dict[str, Any],
        str,
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        return (
            None,
            {},
            "",
            "Select a job row to create a custom resume and cover letter.",
            gr.update(interactive=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )

    def reset_search_save_ui() -> tuple[dict[str, Any], str]:
        return (gr.update(interactive=False), "")

    def select_job_result_ui(
        source_type: str,
        matches_state: list[dict[str, Any]] | None,
        artifacts_state: dict[str, Any] | None,
        evt: gr.SelectData,
    ) -> tuple[int | None, str, str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        matches = matches_from_state(matches_state)
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(matches):
            return (
                None,
                "",
                "Select a job row to create a custom resume and cover letter.",
                gr.update(interactive=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=False),
            )

        match = matches[row_index]
        selected_markdown = _selected_job_markdown(match)
        artifact = (artifacts_state or {}).get(str(row_index), {})
        resume_path = str(artifact.get("resume_pdf_path") or "").strip()
        cover_letter_path = str(artifact.get("cover_letter_path") or "").strip()

        if source_type.strip().lower() != "rxresume":
            return (
                row_index,
                selected_markdown,
                "Custom resume generation is only available when the active resume source is Reactive Resume.",
                gr.update(interactive=False),
                gr.update(value=resume_path or None, visible=bool(resume_path)),
                gr.update(value=cover_letter_path or None, visible=bool(cover_letter_path)),
                gr.update(interactive=True),
            )

        status_text = (
            str(artifact.get("status") or "").strip()
            or f"Ready to generate a custom resume for {match.job.company}."
        )
        return (
            row_index,
            selected_markdown,
            status_text,
            gr.update(interactive=True),
            gr.update(value=resume_path or None, visible=bool(resume_path)),
            gr.update(value=cover_letter_path or None, visible=bool(cover_letter_path)),
            gr.update(interactive=True),
        )

    def create_custom_resume_ui(
        source_type: str,
        rxresume_base_url: str,
        rxresume_resume_id: str,
        candidate_profile: dict[str, Any] | None,
        matches_state: list[dict[str, Any]] | None,
        selected_result_index: int | None,
        artifacts_state: dict[str, Any] | None,
        openai_model: str,
    ) -> tuple[dict[str, Any], pd.DataFrame, str, dict[str, Any], dict[str, Any], str]:
        matches = matches_from_state(matches_state)
        if selected_result_index is None or selected_result_index < 0 or selected_result_index >= len(matches):
            return (
                artifacts_state or {},
                results_frame_from_state(matches_state, artifacts_state),
                "Select a job row before generating a custom resume.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                "",
            )

        result = controller.generate_custom_resume(
            source_type=source_type,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            candidate_profile=candidate_profile,
            match=(matches_state or [])[selected_result_index],
            openai_api_key="",
            openai_model=openai_model,
        )

        updated_artifacts = dict(artifacts_state or {})
        if result.get("resume_pdf_path") and result.get("cover_letter_path"):
            updated_artifacts[str(selected_result_index)] = result

        selected_markdown = _selected_job_markdown(matches[selected_result_index])
        return (
            updated_artifacts,
            results_frame_from_state(matches_state, updated_artifacts),
            result.get("status", ""),
            gr.update(value=result.get("resume_pdf_path") or None, visible=bool(result.get("resume_pdf_path"))),
            gr.update(
                value=result.get("cover_letter_path") or None,
                visible=bool(result.get("cover_letter_path")),
            ),
            selected_markdown,
        )

    def refresh_saved_jobs_ui() -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
        result = controller.list_saved_jobs()
        return (
            saved_jobs_frame_from_state(result["saved_jobs_state"]),
            result["saved_jobs_state"],
            result["status"],
        )

    def save_selected_job_ui(
        matches_state: list[dict[str, Any]] | None,
        selected_result_index: int | None,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], int | None, str, str, dict[str, Any]]:
        if selected_result_index is None:
            result = controller.save_saved_job(None)
        else:
            matches = matches_state or []
            selected_match = matches[selected_result_index] if 0 <= selected_result_index < len(matches) else None
            result = controller.save_saved_job(selected_match)
        return (
            saved_jobs_frame_from_state(result["saved_jobs_state"]),
            result["saved_jobs_state"],
            result.get("selected_saved_job_id"),
            result["status"],
            result["status"],
            gr.update(value=result["saved_jobs_button_label"]),
        )

    def select_saved_job_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        records = saved_jobs_from_state(saved_jobs_state)
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(records):
            return empty_saved_job_editor_ui()
        return saved_job_editor_ui(records[row_index], status_text=f"Editing saved job #{records[row_index].id}.")

    def update_saved_job_ui(
        saved_job_id: int | None,
        provider: str,
        provider_job_id: str,
        title: str,
        company: str,
        location: str,
        pay_range: str,
        via: str,
        apply_url: str,
        share_url: str,
        remote_flag: bool,
        posted_at: str,
        matched_skills_text: str,
        missing_signals_text: str,
        description: str,
        rationale: str,
        score_10: float,
    ) -> tuple[Any, ...]:
        result = controller.update_saved_job(
            saved_job_id=saved_job_id,
            provider=provider,
            provider_job_id=provider_job_id,
            title=title,
            company=company,
            location=location,
            pay_range=pay_range,
            via=via,
            description=description,
            posted_at=posted_at,
            remote_flag=remote_flag,
            apply_url=apply_url,
            share_url=share_url,
            score_10=score_10,
            rationale=rationale,
            matched_skills_text=matched_skills_text,
            missing_signals_text=missing_signals_text,
        )
        saved_job = result.get("saved_job")
        record = SavedJobRecord.model_validate(saved_job) if saved_job else None
        return (
            saved_jobs_frame_from_state(result["saved_jobs_state"]),
            result["saved_jobs_state"],
            gr.update(value=result["saved_jobs_button_label"]),
            *saved_job_editor_ui(record, status_text=result["status"]),
        )

    def delete_saved_job_ui(
        saved_job_id: int | None,
    ) -> tuple[Any, ...]:
        result = controller.delete_saved_job(saved_job_id)
        return (
            saved_jobs_frame_from_state(result["saved_jobs_state"]),
            result["saved_jobs_state"],
            gr.update(value=result["saved_jobs_button_label"]),
            *empty_saved_job_editor_ui(result["status"]),
        )

    def create_saved_job_resume_ui(
        source_type: str,
        rxresume_base_url: str,
        rxresume_resume_id: str,
        candidate_profile: dict[str, Any] | None,
        saved_job_match: dict[str, Any] | None,
        openai_model: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if candidate_profile is None:
            return (
                "Analyze the selected resume before generating a custom resume for a saved job.",
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        result = controller.generate_custom_resume(
            source_type=source_type,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            candidate_profile=candidate_profile,
            match=saved_job_match,
            openai_api_key="",
            openai_model=openai_model,
        )
        return (
            result.get("status", ""),
            gr.update(value=result.get("resume_pdf_path") or None, visible=bool(result.get("resume_pdf_path"))),
            gr.update(
                value=result.get("cover_letter_path") or None,
                visible=bool(result.get("cover_letter_path")),
            ),
        )

    def preview_resume_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_base_url: str,
        rxresume_resume_id: str,
        openai_model: str,
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
            openai_model=openai_model,
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
        openai_model: str,
    ) -> tuple[
        str,
        str,
        str,
        str,
        dict[str, Any] | None,
        pd.DataFrame,
        str,
        dict[str, Any],
        str,
        list[dict[str, Any]],
        None,
        dict[str, Any],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        str,
    ]:
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
                [],
                None,
                {},
                "",
                gr.update(interactive=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
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
            openai_model=openai_model,
        )
        frame = results_frame_from_state(result.get("matches_state") or [])
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
            result.get("matches_state") or [],
            None,
            {},
            "",
            gr.update(interactive=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            "",
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
            setup_openai_model = gr.Textbox(
                label="Backend model",
                value=settings_values["openai_model"],
                placeholder="gpt-5",
                info="Examples: gpt-5, gpt-5.4, gpt-4o",
            )
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
                    gr.Markdown("## Workspace")
                    gr.Markdown("Keys are loaded from `.env` when available. Uploaded PDFs are stored in `.resume`.")
                with gr.Column(scale=2, min_width=420):
                    with gr.Row(elem_id="top-actions"):
                        job_search_nav_button = gr.Button(
                            "Job Search",
                            variant="primary",
                            size="sm",
                            elem_id="job-search-nav-button",
                        )
                        saved_jobs_nav_button = gr.Button(
                            initial_saved_jobs["saved_jobs_button_label"],
                            variant="secondary",
                            size="sm",
                            elem_id="saved-jobs-nav-button",
                        )
                        settings_toggle_button = gr.Button("Settings", size="sm", elem_id="search-settings-toggle")
            settings_panel_visible = gr.State(value=False)
            with gr.Group(visible=False, elem_id="settings-panel") as settings_group:
                settings_openai_api_key = gr.Textbox(
                    label="OpenAI API key",
                    type="password",
                    value=settings_values["openai_api_key"],
                )
                settings_openai_model = gr.Textbox(
                    label="Backend model",
                    value=settings_values["openai_model"],
                    placeholder="gpt-5",
                    info="Examples: gpt-5, gpt-5.4, gpt-4o",
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
            candidate_profile_state = gr.State(value=None)
            analysis_token_state = gr.State(value="")
            matches_state = gr.State(value=[])
            selected_result_index_state = gr.State(value=None)
            generated_artifacts_state = gr.State(value={})
            current_view_state = gr.State(value="search")
            saved_jobs_state = gr.State(value=initial_saved_jobs["saved_jobs_state"])
            selected_saved_job_id_state = gr.State(value=None)
            selected_saved_job_match_state = gr.State(value=None)

            with gr.Group(visible=True) as job_search_workspace:
                gr.Markdown("## Job Search")
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
                results_table = gr.Dataframe(
                    value=_empty_results_frame(),
                    headers=[
                        "score",
                        "title",
                        "company",
                        "location",
                        "pay_range",
                        "via",
                        "apply_url",
                        "custom_resume",
                        "rationale",
                    ],
                    interactive=False,
                    datatype=["number", "str", "str", "str", "str", "str", "html", "str", "str"],
                    wrap=False,
                    row_count=10,
                    max_height=720,
                    show_fullscreen_button=True,
                    column_widths=[80, 300, 220, 220, 180, 180, 320, 160, 420],
                    elem_id="job-results-table",
                )
                results_markdown = gr.Markdown()
                selected_job_markdown = gr.Markdown()
                with gr.Row():
                    save_job_button = gr.Button("Save selected job", variant="secondary", interactive=False)
                    create_custom_resume_button = gr.Button(
                        "Create custom resume and cover letter",
                        variant="secondary",
                        interactive=False,
                    )
                save_job_status = gr.Markdown()
                custom_resume_status = gr.Markdown(
                    "Select a job row to create a custom resume and cover letter."
                )
                with gr.Row():
                    resume_pdf_download = gr.File(label="Tailored resume PDF", visible=False)
                    cover_letter_download = gr.File(label="Cover letter", visible=False)

            with gr.Group(visible=False) as saved_jobs_workspace:
                gr.Markdown("## Saved Jobs")
                saved_jobs_status = gr.Markdown(initial_saved_jobs["status"])
                saved_jobs_table = gr.Dataframe(
                    value=(
                        saved_jobs_frame_from_state(initial_saved_jobs["saved_jobs_state"])
                        if initial_saved_jobs["saved_jobs_state"]
                        else _empty_saved_jobs_frame()
                    ),
                    headers=[
                        "saved_id",
                        "score",
                        "title",
                        "company",
                        "location",
                        "pay_range",
                        "via",
                        "apply_url",
                        "updated_at",
                    ],
                    interactive=False,
                    datatype=["number", "number", "str", "str", "str", "str", "str", "html", "str"],
                    wrap=False,
                    row_count=10,
                    max_height=720,
                    show_fullscreen_button=True,
                    column_widths=[100, 80, 300, 220, 220, 180, 180, 320, 220],
                    elem_id="saved-jobs-table",
                )
                selected_saved_job_markdown = gr.Markdown()
                with gr.Row():
                    saved_job_provider = gr.Textbox(label="Provider")
                    saved_job_score = gr.Number(label="Score", precision=0)
                    saved_job_provider_job_id = gr.Textbox(label="Provider job ID")
                with gr.Row():
                    saved_job_title = gr.Textbox(label="Title")
                    saved_job_company = gr.Textbox(label="Company")
                with gr.Row():
                    saved_job_location = gr.Textbox(label="Location")
                    saved_job_pay_range = gr.Textbox(label="Pay range")
                with gr.Row():
                    saved_job_via = gr.Textbox(label="Via")
                    saved_job_posted_at = gr.Textbox(label="Posted at")
                with gr.Row():
                    saved_job_apply_url = gr.Textbox(label="Apply URL")
                    saved_job_share_url = gr.Textbox(label="Share URL")
                saved_job_remote_flag = gr.Checkbox(label="Remote job", value=False)
                with gr.Row():
                    saved_job_matched_skills = gr.Textbox(
                        label="Matched skills",
                        lines=3,
                        placeholder="One item per line",
                    )
                    saved_job_missing_signals = gr.Textbox(
                        label="Missing signals",
                        lines=3,
                        placeholder="One item per line",
                    )
                saved_job_description = gr.Textbox(label="Description", lines=6)
                saved_job_rationale = gr.Textbox(label="Rationale", lines=4)
                with gr.Row():
                    save_saved_job_changes_button = gr.Button(
                        "Save changes",
                        variant="primary",
                        interactive=False,
                    )
                    delete_saved_job_button = gr.Button(
                        "Delete saved job",
                        variant="stop",
                        interactive=False,
                    )
                    create_saved_job_resume_button = gr.Button(
                        "Create custom resume and cover letter",
                        variant="secondary",
                        interactive=False,
                    )
                saved_job_custom_resume_status = gr.Markdown()
                with gr.Row():
                    saved_resume_pdf_download = gr.File(label="Tailored resume PDF", visible=False)
                    saved_cover_letter_download = gr.File(label="Cover letter", visible=False)

        setup_source_type.change(
            toggle_setup_source,
            inputs=[setup_source_type],
            outputs=[setup_pdf_group],
        )
        save_setup_button.click(
            save_setup_ui,
            inputs=[
                setup_openai_api_key,
                setup_openai_model,
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
                settings_openai_model,
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
        job_search_nav_button.click(
            open_search_view_ui,
            inputs=[saved_jobs_state],
            outputs=[
                current_view_state,
                job_search_workspace,
                saved_jobs_workspace,
                job_search_nav_button,
                saved_jobs_nav_button,
            ],
            queue=False,
        )
        saved_jobs_nav_button.click(
            open_saved_jobs_view_ui,
            inputs=[saved_jobs_state, selected_saved_job_id_state],
            outputs=[
                current_view_state,
                job_search_workspace,
                saved_jobs_workspace,
                job_search_nav_button,
                saved_jobs_nav_button,
                saved_jobs_table,
                selected_saved_job_id_state,
                selected_saved_job_match_state,
                selected_saved_job_markdown,
                saved_jobs_status,
                saved_job_provider,
                saved_job_score,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_posted_at,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                save_saved_job_changes_button,
                delete_saved_job_button,
                create_saved_job_resume_button,
                saved_job_custom_resume_status,
                saved_resume_pdf_download,
                saved_cover_letter_download,
            ],
            queue=False,
        )
        save_settings_button.click(
            save_settings_ui,
            inputs=[
                settings_openai_api_key,
                settings_openai_model,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
            ],
            outputs=[
                settings_status,
                settings_openai_api_key,
                settings_openai_model,
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
        source_type.change(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        source_type.change(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
        uploaded_pdf.change(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        uploaded_pdf.change(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
        saved_pdf_name.change(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        saved_pdf_name.change(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
        load_resumes_button.click(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        load_resumes_button.click(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
        rxresume_base_url.change(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        rxresume_base_url.change(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
        rxresume_resume_id.change(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        rxresume_resume_id.change(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
            queue=False,
        )
        analyze_resume_button.click(
            preview_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_base_url,
                rxresume_resume_id,
                settings_openai_model,
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
        analyze_resume_button.click(
            reset_custom_resume_ui,
            outputs=[
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
            ],
            queue=False,
        )
        analyze_resume_button.click(
            reset_search_save_ui,
            outputs=[save_job_button, save_job_status],
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
                settings_openai_model,
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
                matches_state,
                selected_result_index_state,
                generated_artifacts_state,
                selected_job_markdown,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
                save_job_button,
                save_job_status,
            ],
            queue=False,
        )
        results_table.select(
            select_job_result_ui,
            inputs=[source_type, matches_state, generated_artifacts_state],
            outputs=[
                selected_result_index_state,
                selected_job_markdown,
                custom_resume_status,
                create_custom_resume_button,
                resume_pdf_download,
                cover_letter_download,
                save_job_button,
            ],
            queue=False,
        )
        save_job_button.click(
            save_selected_job_ui,
            inputs=[matches_state, selected_result_index_state],
            outputs=[
                saved_jobs_table,
                saved_jobs_state,
                selected_saved_job_id_state,
                save_job_status,
                saved_jobs_status,
                saved_jobs_nav_button,
            ],
            queue=False,
        )
        create_custom_resume_button.click(
            create_custom_resume_ui,
            inputs=[
                source_type,
                rxresume_base_url,
                rxresume_resume_id,
                candidate_profile_state,
                matches_state,
                selected_result_index_state,
                generated_artifacts_state,
                settings_openai_model,
            ],
            outputs=[
                generated_artifacts_state,
                results_table,
                custom_resume_status,
                resume_pdf_download,
                cover_letter_download,
                selected_job_markdown,
            ],
            queue=False,
        )
        saved_jobs_table.select(
            select_saved_job_ui,
            inputs=[saved_jobs_state],
            outputs=[
                selected_saved_job_id_state,
                selected_saved_job_match_state,
                selected_saved_job_markdown,
                saved_jobs_status,
                saved_job_provider,
                saved_job_score,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_posted_at,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                save_saved_job_changes_button,
                delete_saved_job_button,
                create_saved_job_resume_button,
                saved_job_custom_resume_status,
                saved_resume_pdf_download,
                saved_cover_letter_download,
            ],
            queue=False,
        )
        save_saved_job_changes_button.click(
            update_saved_job_ui,
            inputs=[
                selected_saved_job_id_state,
                saved_job_provider,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_posted_at,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                saved_job_score,
            ],
            outputs=[
                saved_jobs_table,
                saved_jobs_state,
                saved_jobs_nav_button,
                selected_saved_job_id_state,
                selected_saved_job_match_state,
                selected_saved_job_markdown,
                saved_jobs_status,
                saved_job_provider,
                saved_job_score,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_posted_at,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                save_saved_job_changes_button,
                delete_saved_job_button,
                create_saved_job_resume_button,
                saved_job_custom_resume_status,
                saved_resume_pdf_download,
                saved_cover_letter_download,
            ],
            queue=False,
        )
        delete_saved_job_button.click(
            delete_saved_job_ui,
            inputs=[selected_saved_job_id_state],
            outputs=[
                saved_jobs_table,
                saved_jobs_state,
                saved_jobs_nav_button,
                selected_saved_job_id_state,
                selected_saved_job_match_state,
                selected_saved_job_markdown,
                saved_jobs_status,
                saved_job_provider,
                saved_job_score,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_posted_at,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                save_saved_job_changes_button,
                delete_saved_job_button,
                create_saved_job_resume_button,
                saved_job_custom_resume_status,
                saved_resume_pdf_download,
                saved_cover_letter_download,
            ],
            queue=False,
        )
        create_saved_job_resume_button.click(
            create_saved_job_resume_ui,
            inputs=[
                source_type,
                rxresume_base_url,
                rxresume_resume_id,
                candidate_profile_state,
                selected_saved_job_match_state,
                settings_openai_model,
            ],
            outputs=[
                saved_job_custom_resume_status,
                saved_resume_pdf_download,
                saved_cover_letter_download,
            ],
            queue=False,
        )

    return demo


def build_app(
    controller: AppController | None = None,
    *,
    workspace: LocalWorkspace | None = None,
) -> gr.Blocks:
    workspace = workspace or LocalWorkspace()
    controller = controller or AppController(workspace=workspace)

    settings_values = controller.current_settings()
    initial_saved_jobs = controller.list_saved_jobs()
    initial_source = "pdf"

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

    def current_resume_token(source_type: str, saved_resume_name: str | None, rxresume_resume_id: str) -> str:
        source = source_type.strip().lower()
        if source == "pdf":
            return f"pdf:{(saved_resume_name or '').strip()}"
        if source == "rxresume":
            return f"rxresume:{(rxresume_resume_id or '').strip()}"
        return source

    def normalize_rxresume_options(options_state: list[Any] | None) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for option in options_state or []:
            if isinstance(option, (list, tuple)) and len(option) >= 2:
                normalized.append((str(option[0]), str(option[1])))
        return normalized

    def matches_from_state(matches_state: list[dict[str, Any]] | None) -> list[ScoredJobMatch]:
        return [ScoredJobMatch.model_validate(match) for match in (matches_state or [])]

    def saved_jobs_from_state(saved_jobs_state: list[dict[str, Any]] | None) -> list[SavedJobRecord]:
        return [SavedJobRecord.model_validate(saved_job) for saved_job in (saved_jobs_state or [])]

    def saved_jobs_frame_from_state(saved_jobs_state: list[dict[str, Any]] | None) -> pd.DataFrame:
        rows = _rows_from_saved_jobs(saved_jobs_from_state(saved_jobs_state), linkify_apply_url=True)
        return pd.DataFrame(rows) if rows else _empty_saved_jobs_frame()

    def source_is_ready(source_type: str, saved_resume_name: str | None, rxresume_resume_id: str) -> bool:
        if source_type.strip().lower() == "pdf":
            return bool((saved_resume_name or "").strip())
        return bool((rxresume_resume_id or "").strip())

    def default_search_status(source_type: str, saved_resume_name: str | None, rxresume_resume_id: str) -> str:
        if source_is_ready(source_type, saved_resume_name, rxresume_resume_id):
            return "Analyze the selected resume to continue."
        if source_type.strip().lower() == "rxresume":
            return "Load your Reactive Resume entries and choose one to continue."
        return "Choose or upload a PDF resume to get started."

    def default_source_draft(source_type: str) -> dict[str, Any]:
        saved_resume_name = controller.default_saved_resume_name() if source_type == "pdf" else None
        rxresume_resume_id = ""
        return {
            "saved_resume_name": saved_resume_name,
            "rxresume_resume_id": rxresume_resume_id,
            "rxresume_options_state": [],
            "location_override": "",
            "include_remote": True,
            "search_terms_text": "",
            "candidate_profile": None,
            "analysis_token": "",
            "matches_state": [],
            "search_attempted": False,
            "generated_artifacts_state": {},
            "sort_by": "Best match",
            "filter_text": "",
            "selected_result_match": None,
            "status_text": default_search_status(source_type, saved_resume_name, rxresume_resume_id),
        }

    def normalize_source_drafts(drafts_state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        drafts = {"pdf": default_source_draft("pdf"), "rxresume": default_source_draft("rxresume")}
        if isinstance(drafts_state, dict):
            for source_name in ("pdf", "rxresume"):
                current = drafts_state.get(source_name)
                if isinstance(current, dict):
                    drafts[source_name].update(current)
        return drafts

    def saved_job_record_by_id(
        saved_jobs_state: list[dict[str, Any]] | None,
        saved_job_id: int | None,
    ) -> SavedJobRecord | None:
        if saved_job_id is None:
            return None
        for record in saved_jobs_from_state(saved_jobs_state):
            if record.id == saved_job_id:
                return record
        return None

    def current_results_frame(
        matches_state: list[dict[str, Any]] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        sort_by: str,
        filter_text: str,
    ) -> pd.DataFrame:
        visible_matches = _filter_and_sort_matches(
            matches_from_state(matches_state),
            sort_by=sort_by or "Best match",
            filter_text=filter_text or "",
        )
        rows = _rows_from_matches(
            visible_matches,
            linkify_apply_url=True,
            saved_jobs_state=saved_jobs_state,
        )
        return pd.DataFrame(rows) if rows else _empty_results_frame()

    def search_workspace_response(
        *,
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        selected_result_match: dict[str, Any] | None,
        status_text: str,
    ) -> tuple[Any, ...]:
        source = source_type.strip().lower() or "pdf"
        rxresume_options = normalize_rxresume_options(rxresume_options_state)
        current_token = current_resume_token(source, saved_resume_name, rxresume_resume_id)
        source_ready = source_is_ready(source, saved_resume_name, rxresume_resume_id)
        analysis_complete = candidate_profile is not None and analysis_token == current_token
        raw_matches = matches_from_state(matches_state)
        visible_matches = _filter_and_sort_matches(
            raw_matches,
            sort_by=sort_by or "Best match",
            filter_text=filter_text or "",
        )
        visible_matches_state = [match.model_dump() for match in visible_matches]

        selected_match: ScoredJobMatch | None = None
        if selected_result_match is not None:
            try:
                candidate_match = ScoredJobMatch.model_validate(selected_result_match)
                candidate_key = saved_job_identity(candidate_match)
                for visible_match in visible_matches:
                    if saved_job_identity(visible_match) == candidate_key:
                        selected_match = visible_match
                        break
            except Exception:
                selected_match = None

        if status_text.strip():
            resolved_status = status_text.strip()
        elif search_attempted and raw_matches and filter_text.strip() and not visible_matches:
            resolved_status = f'No results match "{filter_text.strip()}".'
        elif search_attempted:
            resolved_status = "No jobs met the 7/10 threshold." if not raw_matches else f"Found {len(raw_matches)} matching jobs."
        elif analysis_complete:
            resolved_status = "Adjust the search preferences, then click Find jobs."
        else:
            resolved_status = default_search_status(source, saved_resume_name, rxresume_resume_id)

        source_state = "Complete" if source_ready else "Current"
        if analysis_complete:
            analyze_state = "Complete"
        elif source_ready:
            analyze_state = "Current"
        else:
            analyze_state = "Ready"
        if search_attempted:
            preferences_state = "Complete"
            results_state = "Complete"
        elif analysis_complete:
            preferences_state = "Current"
            results_state = "Ready"
        else:
            preferences_state = "Ready"
            results_state = "Ready"

        location_used = ""
        if candidate_profile is not None:
            profile = CandidateProfile.model_validate(candidate_profile)
            location_used = location_override.strip() or profile.inferred_location

        results_frame = current_results_frame(matches_state, saved_jobs_state, sort_by, filter_text)

        artifact = {}
        if selected_match is not None:
            artifact = dict((generated_artifacts_state or {}).get(saved_job_identity(selected_match), {}))
        resume_path = str(artifact.get("resume_pdf_path") or "").strip()
        cover_letter_path = str(artifact.get("cover_letter_path") or "").strip()
        selected_visible = selected_match is not None

        return (
            gr.update(visible=source == "pdf"),
            gr.update(visible=source == "rxresume"),
            saved_resume_update(saved_resume_name),
            gr.update(choices=rxresume_options, value=rxresume_resume_id or None),
            rxresume_options,
            _resume_source_summary_html(
                source_type=source,
                saved_resume_name=saved_resume_name,
                rxresume_resume_id=rxresume_resume_id,
                rxresume_options=rxresume_options,
            ),
            _candidate_summary_html(candidate_profile, location_used=location_used),
            _status_html(resolved_status),
            resolved_status,
            _step_header_html(1, "Resume Source", source_state),
            _step_header_html(2, "Analyze Resume", analyze_state),
            _step_header_html(3, "Search Preferences", preferences_state),
            _step_header_html(4, "Results", results_state),
            location_override,
            include_remote,
            search_terms_text,
            sort_by or "Best match",
            filter_text,
            candidate_profile,
            analysis_token,
            gr.update(interactive=analysis_complete),
            matches_state or [],
            visible_matches_state,
            bool(search_attempted),
            dict(generated_artifacts_state or {}),
            results_frame,
            selected_match.model_dump() if selected_match is not None else None,
            gr.update(visible=selected_visible),
            _job_detail_html(selected_match) if selected_visible else "",
            gr.update(visible=selected_visible),
            gr.update(interactive=selected_visible),
            gr.update(interactive=selected_visible and source == "rxresume"),
            gr.update(value=resume_path or None, visible=bool(resume_path)),
            gr.update(value=cover_letter_path or None, visible=bool(cover_letter_path)),
        )

    def saved_jobs_workspace_response(
        *,
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
        status_text: str,
        edit_mode: bool,
        delete_confirm: bool,
        resume_pdf_path: str = "",
        cover_letter_path: str = "",
    ) -> tuple[Any, ...]:
        records = saved_jobs_from_state(saved_jobs_state)
        record = saved_job_record_by_id(saved_jobs_state, selected_saved_job_id)
        current_edit_mode = bool(edit_mode and record is not None)
        current_delete_confirm = bool(delete_confirm and record is not None and not current_edit_mode)
        if status_text.strip():
            resolved_status = status_text.strip()
        elif records:
            resolved_status = "Select a saved job to review."
        else:
            resolved_status = "No saved jobs yet."
        return (
            gr.update(label=_saved_jobs_tab_label(len(records))),
            saved_jobs_state or [],
            saved_jobs_frame_from_state(saved_jobs_state),
            _status_html(resolved_status),
            resolved_status,
            record.id if record is not None else None,
            record.match.model_dump() if record is not None else None,
            gr.update(visible=record is not None and not current_edit_mode),
            _saved_job_summary_html(record) if record is not None and not current_edit_mode else "",
            _apply_link_button_html(record.match.job.apply_url) if record is not None and not current_edit_mode else "",
            gr.update(visible=record is not None and not current_edit_mode),
            gr.update(interactive=record is not None and not current_edit_mode),
            gr.update(
                value="Confirm Delete" if current_delete_confirm else "Delete",
                interactive=record is not None and not current_edit_mode,
            ),
            gr.update(visible=current_delete_confirm and record is not None and not current_edit_mode),
            gr.update(interactive=record is not None and not current_edit_mode),
            gr.update(visible=current_edit_mode),
            record.match.job.provider if record is not None else "",
            record.match.job.provider_job_id if record is not None else "",
            record.match.job.title if record is not None else "",
            record.match.job.company if record is not None else "",
            record.match.job.location if record is not None else "",
            record.match.job.pay_range if record is not None else "",
            record.match.job.via if record is not None else "",
            record.match.job.posted_at if record is not None else "",
            record.match.job.apply_url if record is not None else "",
            record.match.job.share_url if record is not None else "",
            record.match.job.remote_flag if record is not None else False,
            float(record.match.score_10) if record is not None else 0,
            _list_text(record.match.matched_skills) if record is not None else "",
            _list_text(record.match.missing_signals) if record is not None else "",
            record.match.job.description if record is not None else "",
            record.match.rationale if record is not None else "",
            gr.update(value=resume_pdf_path or None, visible=bool(resume_pdf_path)),
            gr.update(value=cover_letter_path or None, visible=bool(cover_letter_path)),
            current_edit_mode,
            current_delete_confirm,
        )

    initial_drafts = normalize_source_drafts(None)
    initial_pdf_draft = initial_drafts["pdf"]

    def validate_setup_inputs(
        openai_api_key: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        source_type: str,
        pdf_file: str | None,
    ) -> tuple[dict[str, Any], str]:
        messages: list[str] = []
        if not openai_api_key.strip():
            messages.append("OpenAI API key is required.")
        if not serpapi_api_key.strip():
            messages.append("SerpApi key is required.")
        if source_type.strip().lower() == "rxresume" and not rxresume_api_key.strip():
            messages.append("Reactive Resume API key is required for this source.")
        if source_type.strip().lower() == "pdf" and not pdf_file:
            messages.append("Upload a PDF resume to continue.")
        ready = not messages
        message = "All required setup fields are ready." if ready else " ".join(messages)
        return gr.update(interactive=ready), _status_html(message)

    def toggle_setup_source(source_type: str) -> dict[str, Any]:
        return gr.update(visible=source_type.strip().lower() == "pdf")

    def toggle_settings_ui(current_visible: bool) -> tuple[bool, dict[str, Any]]:
        next_visible = not current_visible
        return next_visible, gr.update(visible=next_visible)

    def reset_current_resume_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        include_remote: bool,
        sort_by: str,
        filter_text: str,
        status_text: str,
    ) -> tuple[Any, ...]:
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override="",
            include_remote=include_remote,
            search_terms_text="",
            candidate_profile=None,
            analysis_token="",
            matches_state=[],
            search_attempted=False,
            generated_artifacts_state={},
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=None,
            status_text=status_text,
        )

    def save_setup_ui(
        openai_api_key: str,
        openai_model: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
        source_type: str,
        setup_pdf_file: str | None,
    ) -> tuple[Any, ...]:
        try:
            result = controller.complete_setup(
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                serpapi_api_key=serpapi_api_key,
                rxresume_api_key=rxresume_api_key,
                rxresume_api_url=rxresume_api_url,
                source_type=source_type,
                pdf_file=setup_pdf_file,
            )
        except Exception as exc:
            return (
                _status_html(str(exc)),
                gr.update(visible=True),
                gr.update(visible=False),
                source_type,
                initial_source,
                initial_drafts,
                openai_api_key,
                openai_model or DEFAULT_OPENAI_MODEL,
                serpapi_api_key,
                rxresume_api_key,
                rxresume_api_url or controller.default_rxresume_api_url(),
                *search_workspace_response(
                    source_type=initial_source,
                    saved_resume_name=initial_pdf_draft["saved_resume_name"],
                    rxresume_resume_id=initial_pdf_draft["rxresume_resume_id"],
                    rxresume_options_state=initial_pdf_draft["rxresume_options_state"],
                    saved_jobs_state=initial_saved_jobs["saved_jobs_state"],
                    location_override=initial_pdf_draft["location_override"],
                    include_remote=initial_pdf_draft["include_remote"],
                    search_terms_text=initial_pdf_draft["search_terms_text"],
                    candidate_profile=initial_pdf_draft["candidate_profile"],
                    analysis_token=initial_pdf_draft["analysis_token"],
                    matches_state=initial_pdf_draft["matches_state"],
                    search_attempted=initial_pdf_draft["search_attempted"],
                    generated_artifacts_state=initial_pdf_draft["generated_artifacts_state"],
                    sort_by=initial_pdf_draft["sort_by"],
                    filter_text=initial_pdf_draft["filter_text"],
                    selected_result_match=initial_pdf_draft["selected_result_match"],
                    status_text=initial_pdf_draft["status_text"],
                ),
            )

        selected_source = result["source_type"]
        settings_after_save = controller.current_settings()
        drafts = normalize_source_drafts(None)
        drafts[selected_source]["saved_resume_name"] = result["saved_resume_name"]
        drafts[selected_source]["status_text"] = default_search_status(selected_source, result["saved_resume_name"], "")

        return (
            _status_html(result["status"]),
            gr.update(visible=False),
            gr.update(visible=True),
            selected_source,
            selected_source,
            drafts,
            settings_after_save["openai_api_key"],
            settings_after_save["openai_model"],
            settings_after_save["serpapi_api_key"],
            settings_after_save["rxresume_api_key"],
            settings_after_save["rxresume_api_url"],
            *search_workspace_response(
                source_type=selected_source,
                saved_resume_name=drafts[selected_source]["saved_resume_name"],
                rxresume_resume_id=drafts[selected_source]["rxresume_resume_id"],
                rxresume_options_state=drafts[selected_source]["rxresume_options_state"],
                saved_jobs_state=initial_saved_jobs["saved_jobs_state"],
                location_override="",
                include_remote=True,
                search_terms_text="",
                candidate_profile=None,
                analysis_token="",
                matches_state=[],
                search_attempted=False,
                generated_artifacts_state={},
                sort_by="Best match",
                filter_text="",
                selected_result_match=None,
                status_text=drafts[selected_source]["status_text"],
            ),
        )

    def save_settings_ui(
        openai_api_key: str,
        openai_model: str,
        serpapi_api_key: str,
        rxresume_api_key: str,
        rxresume_api_url: str,
    ) -> tuple[str, str, str, str, str, str]:
        try:
            result = controller.save_settings(
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                serpapi_api_key=serpapi_api_key,
                rxresume_api_key=rxresume_api_key,
                rxresume_api_url=rxresume_api_url,
            )
        except Exception as exc:
            return (
                _status_html(str(exc)),
                openai_api_key,
                openai_model or DEFAULT_OPENAI_MODEL,
                serpapi_api_key,
                rxresume_api_key,
                rxresume_api_url or controller.default_rxresume_api_url(),
            )
        return (
            _status_html(result["status"]),
            result["openai_api_key"],
            result["openai_model"],
            result["serpapi_api_key"],
            result["rxresume_api_key"],
            result["rxresume_api_url"],
        )

    def switch_source_ui(
        new_source: str,
        active_source: str,
        drafts_state: dict[str, Any] | None,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        selected_result_match: dict[str, Any] | None,
        search_status_text: str,
        saved_jobs_state: list[dict[str, Any]] | None,
    ) -> tuple[Any, ...]:
        drafts = normalize_source_drafts(drafts_state)
        drafts[active_source] = {
            "saved_resume_name": saved_resume_name,
            "rxresume_resume_id": rxresume_resume_id,
            "rxresume_options_state": normalize_rxresume_options(rxresume_options_state),
            "location_override": location_override,
            "include_remote": include_remote,
            "search_terms_text": search_terms_text,
            "candidate_profile": candidate_profile,
            "analysis_token": analysis_token,
            "matches_state": matches_state or [],
            "search_attempted": bool(search_attempted),
            "generated_artifacts_state": dict(generated_artifacts_state or {}),
            "sort_by": sort_by,
            "filter_text": filter_text,
            "selected_result_match": selected_result_match,
            "status_text": search_status_text,
        }
        target = drafts[new_source]
        return (
            new_source,
            drafts,
            *search_workspace_response(
                source_type=new_source,
                saved_resume_name=target["saved_resume_name"],
                rxresume_resume_id=target["rxresume_resume_id"],
                rxresume_options_state=target["rxresume_options_state"],
                saved_jobs_state=saved_jobs_state,
                location_override=target["location_override"],
                include_remote=bool(target.get("include_remote", True)),
                search_terms_text=target["search_terms_text"],
                candidate_profile=target["candidate_profile"],
                analysis_token=target["analysis_token"],
                matches_state=target["matches_state"],
                search_attempted=bool(target["search_attempted"]),
                generated_artifacts_state=target["generated_artifacts_state"],
                sort_by=target["sort_by"],
                filter_text=target["filter_text"],
                selected_result_match=target["selected_result_match"],
                status_text=target["status_text"],
            ),
        )

    def store_uploaded_pdf_ui(
        uploaded_pdf: str | None,
        source_type: str,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        include_remote: bool,
        sort_by: str,
        filter_text: str,
    ) -> tuple[Any, ...]:
        if not uploaded_pdf:
            return reset_current_resume_ui(
                source_type=source_type,
                saved_resume_name=controller.default_saved_resume_name(),
                rxresume_resume_id=rxresume_resume_id,
                rxresume_options_state=rxresume_options_state,
                saved_jobs_state=saved_jobs_state,
                include_remote=include_remote,
                sort_by=sort_by,
                filter_text=filter_text,
                status_text=default_search_status("pdf", controller.default_saved_resume_name(), ""),
            )
        saved_resume = workspace.save_uploaded_resume(uploaded_pdf)
        return reset_current_resume_ui(
            source_type="pdf",
            saved_resume_name=saved_resume.name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            include_remote=include_remote,
            sort_by=sort_by,
            filter_text=filter_text,
            status_text=f"Saved {saved_resume.name}. Analyze the selected resume to continue.",
        )

    def load_resumes_ui(
        source_type: str,
        saved_resume_name: str | None,
        base_url: str,
        saved_jobs_state: list[dict[str, Any]] | None,
        include_remote: bool,
        sort_by: str,
        filter_text: str,
    ) -> tuple[Any, ...]:
        status, options = controller.load_rxresume_options(base_url)
        selected_resume_id = options[0][1] if options else ""
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=selected_resume_id,
            rxresume_options_state=options,
            saved_jobs_state=saved_jobs_state,
            location_override="",
            include_remote=include_remote,
            search_terms_text="",
            candidate_profile=None,
            analysis_token="",
            matches_state=[],
            search_attempted=False,
            generated_artifacts_state={},
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=None,
            status_text=status,
        )

    def preview_resume_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        include_remote: bool,
        sort_by: str,
        filter_text: str,
        openai_model: str,
        rxresume_base_url: str,
    ) -> tuple[Any, ...]:
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
            openai_model=openai_model,
        )
        current_token = ""
        if result["candidate_profile"] is not None:
            current_token = current_resume_token(source_type, saved_resume_name, rxresume_resume_id)
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override=result["location_used"],
            include_remote=include_remote,
            search_terms_text=result["search_terms_text"],
            candidate_profile=result["candidate_profile"],
            analysis_token=current_token,
            matches_state=[],
            search_attempted=False,
            generated_artifacts_state={},
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=None,
            status_text=result["status"],
        )

    def find_jobs_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        sort_by: str,
        filter_text: str,
        openai_model: str,
        rxresume_base_url: str,
    ) -> tuple[Any, ...]:
        current_token = current_resume_token(source_type, saved_resume_name, rxresume_resume_id)
        if candidate_profile is None or not analysis_token or analysis_token != current_token:
            return search_workspace_response(
                source_type=source_type,
                saved_resume_name=saved_resume_name,
                rxresume_resume_id=rxresume_resume_id,
                rxresume_options_state=rxresume_options_state,
                saved_jobs_state=saved_jobs_state,
                location_override=location_override,
                include_remote=include_remote,
                search_terms_text=search_terms_text,
                candidate_profile=None,
                analysis_token="",
                matches_state=[],
                search_attempted=False,
                generated_artifacts_state={},
                sort_by=sort_by,
                filter_text=filter_text,
                selected_result_match=None,
                status_text="Analyze the selected resume before searching.",
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
            openai_model=openai_model,
        )
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override=result["location_used"],
            include_remote=include_remote,
            search_terms_text=result["search_terms_text"],
            candidate_profile=result["candidate_profile"],
            analysis_token=current_token,
            matches_state=result.get("matches_state") or [],
            search_attempted=True,
            generated_artifacts_state={},
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=None,
            status_text=result["status"],
        )

    def refresh_results_controls_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        search_status_text: str,
    ) -> tuple[Any, ...]:
        status_text = search_status_text
        if search_attempted and filter_text.strip():
            raw_matches = matches_from_state(matches_state)
            visible_matches = _filter_and_sort_matches(raw_matches, sort_by=sort_by, filter_text=filter_text)
            status_text = f"Showing {len(visible_matches)} of {len(raw_matches)} jobs."
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override=location_override,
            include_remote=include_remote,
            search_terms_text=search_terms_text,
            candidate_profile=candidate_profile,
            analysis_token=analysis_token,
            matches_state=matches_state,
            search_attempted=search_attempted,
            generated_artifacts_state=generated_artifacts_state,
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=None,
            status_text=status_text,
        )

    def select_job_result_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        search_status_text: str,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        visible_matches = _filter_and_sort_matches(matches_from_state(matches_state), sort_by=sort_by, filter_text=filter_text)
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected_match = None
        if isinstance(row_index, int) and 0 <= row_index < len(visible_matches):
            selected_match = visible_matches[row_index].model_dump()
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override=location_override,
            include_remote=include_remote,
            search_terms_text=search_terms_text,
            candidate_profile=candidate_profile,
            analysis_token=analysis_token,
            matches_state=matches_state,
            search_attempted=search_attempted,
            generated_artifacts_state=generated_artifacts_state,
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=selected_match,
            status_text=search_status_text if selected_match is not None else "Select a job to review its details.",
        )

    def save_selected_job_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        selected_result_match: dict[str, Any] | None,
    ) -> tuple[Any, ...]:
        result = controller.save_saved_job(selected_result_match)
        return (
            *search_workspace_response(
                source_type=source_type,
                saved_resume_name=saved_resume_name,
                rxresume_resume_id=rxresume_resume_id,
                rxresume_options_state=rxresume_options_state,
                saved_jobs_state=result["saved_jobs_state"],
                location_override=location_override,
                include_remote=include_remote,
                search_terms_text=search_terms_text,
                candidate_profile=candidate_profile,
                analysis_token=analysis_token,
                matches_state=matches_state,
                search_attempted=search_attempted,
                generated_artifacts_state=generated_artifacts_state,
                sort_by=sort_by,
                filter_text=filter_text,
                selected_result_match=selected_result_match,
                status_text=result["status"],
            ),
            *saved_jobs_workspace_response(
                saved_jobs_state=result["saved_jobs_state"],
                selected_saved_job_id=result.get("selected_saved_job_id"),
                status_text=result["status"],
                edit_mode=False,
                delete_confirm=False,
            ),
        )

    def create_custom_resume_ui(
        source_type: str,
        saved_resume_name: str | None,
        rxresume_resume_id: str,
        rxresume_options_state: list[Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        location_override: str,
        include_remote: bool,
        search_terms_text: str,
        candidate_profile: dict[str, Any] | None,
        analysis_token: str,
        matches_state: list[dict[str, Any]] | None,
        search_attempted: bool,
        generated_artifacts_state: dict[str, Any] | None,
        sort_by: str,
        filter_text: str,
        selected_result_match: dict[str, Any] | None,
        openai_model: str,
        rxresume_base_url: str,
    ) -> tuple[Any, ...]:
        result = controller.generate_custom_resume(
            source_type=source_type,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            candidate_profile=candidate_profile,
            match=selected_result_match,
            openai_api_key="",
            openai_model=openai_model,
        )
        updated_artifacts = dict(generated_artifacts_state or {})
        if selected_result_match is not None and result.get("resume_pdf_path") and result.get("cover_letter_path"):
            updated_artifacts[saved_job_identity(ScoredJobMatch.model_validate(selected_result_match))] = result
        return search_workspace_response(
            source_type=source_type,
            saved_resume_name=saved_resume_name,
            rxresume_resume_id=rxresume_resume_id,
            rxresume_options_state=rxresume_options_state,
            saved_jobs_state=saved_jobs_state,
            location_override=location_override,
            include_remote=include_remote,
            search_terms_text=search_terms_text,
            candidate_profile=candidate_profile,
            analysis_token=analysis_token,
            matches_state=matches_state,
            search_attempted=search_attempted,
            generated_artifacts_state=updated_artifacts,
            sort_by=sort_by,
            filter_text=filter_text,
            selected_result_match=selected_result_match,
            status_text=result.get("status", ""),
        )

    def open_saved_jobs_tab_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
        saved_jobs_status_text: str,
    ) -> tuple[Any, ...]:
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text=saved_jobs_status_text,
            edit_mode=False,
            delete_confirm=False,
        )

    def select_saved_job_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        records = saved_jobs_from_state(saved_jobs_state)
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected_saved_job_id = records[row_index].id if isinstance(row_index, int) and 0 <= row_index < len(records) else None
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text="Saved job selected.",
            edit_mode=False,
            delete_confirm=False,
        )

    def enter_saved_job_edit_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
    ) -> tuple[Any, ...]:
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text="Edit the saved job details, then save your changes.",
            edit_mode=True,
            delete_confirm=False,
        )

    def cancel_saved_job_edit_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
        saved_jobs_status_text: str,
    ) -> tuple[Any, ...]:
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text=saved_jobs_status_text or "Edit cancelled.",
            edit_mode=False,
            delete_confirm=False,
        )

    def toggle_delete_saved_job_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
        delete_confirm: bool,
        matches_state: list[dict[str, Any]] | None,
        sort_by: str,
        filter_text: str,
    ) -> tuple[Any, ...]:
        if selected_saved_job_id is None:
            saved_jobs_response = saved_jobs_workspace_response(
                saved_jobs_state=saved_jobs_state,
                selected_saved_job_id=None,
                status_text="Select a saved job before deleting.",
                edit_mode=False,
                delete_confirm=False,
            )
            return (*saved_jobs_response, current_results_frame(matches_state, saved_jobs_state, sort_by, filter_text))
        if not delete_confirm:
            saved_jobs_response = saved_jobs_workspace_response(
                saved_jobs_state=saved_jobs_state,
                selected_saved_job_id=selected_saved_job_id,
                status_text="Click Confirm Delete again to remove this saved job.",
                edit_mode=False,
                delete_confirm=True,
            )
            return (*saved_jobs_response, current_results_frame(matches_state, saved_jobs_state, sort_by, filter_text))
        result = controller.delete_saved_job(selected_saved_job_id)
        saved_jobs_response = saved_jobs_workspace_response(
            saved_jobs_state=result["saved_jobs_state"],
            selected_saved_job_id=None,
            status_text=result["status"],
            edit_mode=False,
            delete_confirm=False,
        )
        return (*saved_jobs_response, current_results_frame(matches_state, result["saved_jobs_state"], sort_by, filter_text))

    def cancel_delete_saved_job_ui(
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
    ) -> tuple[Any, ...]:
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text="Delete cancelled.",
            edit_mode=False,
            delete_confirm=False,
        )

    def update_saved_job_ui(
        saved_job_id: int | None,
        provider: str,
        provider_job_id: str,
        title: str,
        company: str,
        location: str,
        pay_range: str,
        via: str,
        posted_at: str,
        apply_url: str,
        share_url: str,
        remote_flag: bool,
        score_10: float,
        matched_skills_text: str,
        missing_signals_text: str,
        description: str,
        rationale: str,
        matches_state: list[dict[str, Any]] | None,
        sort_by: str,
        filter_text: str,
    ) -> tuple[Any, ...]:
        result = controller.update_saved_job(
            saved_job_id=saved_job_id,
            provider=provider,
            provider_job_id=provider_job_id,
            title=title,
            company=company,
            location=location,
            pay_range=pay_range,
            via=via,
            description=description,
            posted_at=posted_at,
            remote_flag=remote_flag,
            apply_url=apply_url,
            share_url=share_url,
            score_10=score_10,
            rationale=rationale,
            matched_skills_text=matched_skills_text,
            missing_signals_text=missing_signals_text,
        )
        saved_jobs_response = saved_jobs_workspace_response(
            saved_jobs_state=result["saved_jobs_state"],
            selected_saved_job_id=result.get("selected_saved_job_id"),
            status_text=result["status"],
            edit_mode=False,
            delete_confirm=False,
        )
        return (*saved_jobs_response, current_results_frame(matches_state, result["saved_jobs_state"], sort_by, filter_text))

    def create_saved_job_resume_ui(
        source_type: str,
        rxresume_resume_id: str,
        candidate_profile: dict[str, Any] | None,
        saved_jobs_state: list[dict[str, Any]] | None,
        selected_saved_job_id: int | None,
        selected_saved_job_match: dict[str, Any] | None,
        openai_model: str,
        rxresume_base_url: str,
    ) -> tuple[Any, ...]:
        result = controller.generate_custom_resume(
            source_type=source_type,
            rxresume_base_url=rxresume_base_url,
            rxresume_api_key="",
            rxresume_resume_id=rxresume_resume_id or "",
            candidate_profile=candidate_profile,
            match=selected_saved_job_match,
            openai_api_key="",
            openai_model=openai_model,
        )
        return saved_jobs_workspace_response(
            saved_jobs_state=saved_jobs_state,
            selected_saved_job_id=selected_saved_job_id,
            status_text=result.get("status", ""),
            edit_mode=False,
            delete_confirm=False,
            resume_pdf_path=result.get("resume_pdf_path", ""),
            cover_letter_path=result.get("cover_letter_path", ""),
        )

    with gr.Blocks(title="Resume to Jobs Finder", css=APP_CSS) as demo:
        with gr.Group(visible=controller.setup_required()) as setup_group:
            gr.Markdown("# Resume to Jobs Finder")
            gr.Markdown('<p class="setup-copy">Save your settings once, choose a resume source, and then move straight into job search.</p>')
            setup_status = gr.HTML(_status_html("Provide the required keys and a starting resume source to continue."))
            with gr.Accordion("AI Configuration", open=True):
                setup_openai_api_key = gr.Textbox(label="OpenAI API key", type="password")
                setup_openai_model = gr.Textbox(
                    label="Backend model",
                    value=settings_values["openai_model"],
                    placeholder="gpt-5",
                    info="Examples: gpt-5, gpt-5.4, gpt-4o",
                )
            with gr.Accordion("Job Search API", open=True):
                setup_serpapi_api_key = gr.Textbox(label="SerpApi key", type="password")
            with gr.Accordion("Resume Platform", open=True):
                setup_source_type = gr.Radio(
                    label="Initial resume source",
                    choices=[("PDF", "pdf"), ("Reactive Resume", "rxresume")],
                    value="pdf",
                )
                setup_rxresume_api_key = gr.Textbox(label="Reactive Resume API key", type="password")
                setup_rxresume_api_url = gr.Textbox(
                    label="Reactive Resume API URL",
                    value=controller.default_rxresume_api_url(),
                    placeholder=DEFAULT_RXRESUME_RESUMES_URL,
                )
                with gr.Group(visible=True) as setup_pdf_group:
                    setup_pdf_file = gr.File(label="Upload PDF resume", file_types=[".pdf"], type="filepath")
            save_setup_button = gr.Button("Save setup and continue", variant="primary", interactive=False)

        with gr.Group(visible=not controller.setup_required()) as search_group:
            with gr.Row(elem_id="app-header"):
                with gr.Column(scale=6):
                    gr.Markdown("# Resume to Jobs Finder")
                    gr.Markdown(
                        '<p class="workspace-copy">Your settings are saved between sessions. Analyze a resume, tune the search, and save promising roles as you go.</p>'
                    )
                with gr.Column(scale=2, min_width=220):
                    with gr.Row(elem_id="top-actions"):
                        settings_toggle_button = gr.Button("Settings", size="sm", elem_id="search-settings-toggle")

            settings_panel_visible = gr.State(value=False)
            with gr.Group(visible=False, elem_id="settings-panel") as settings_group:
                settings_status = gr.HTML("")
                with gr.Accordion("AI Configuration", open=True):
                    settings_openai_api_key = gr.Textbox(
                        label="OpenAI API key",
                        type="password",
                        value=settings_values["openai_api_key"],
                    )
                    settings_openai_model = gr.Textbox(
                        label="Backend model",
                        value=settings_values["openai_model"],
                        placeholder="gpt-5",
                        info="Examples: gpt-5, gpt-5.4, gpt-4o",
                    )
                with gr.Accordion("Job Search API", open=True):
                    settings_serpapi_api_key = gr.Textbox(
                        label="SerpApi key",
                        type="password",
                        value=settings_values["serpapi_api_key"],
                    )
                with gr.Accordion("Resume Platform", open=True):
                    settings_rxresume_api_key = gr.Textbox(
                        label="Reactive Resume API key",
                        type="password",
                        value=settings_values["rxresume_api_key"],
                    )
                    settings_rxresume_api_url = gr.Textbox(
                        label="Reactive Resume API URL",
                        value=settings_values["rxresume_api_url"],
                        placeholder=DEFAULT_RXRESUME_RESUMES_URL,
                    )
                save_settings_button = gr.Button("Save settings", variant="primary")

            active_source_state = gr.State(value=initial_source)
            source_drafts_state = gr.State(value=initial_drafts)
            search_status_text_state = gr.State(value=initial_pdf_draft["status_text"])
            saved_jobs_state = gr.State(value=initial_saved_jobs["saved_jobs_state"])
            saved_jobs_status_text_state = gr.State(value=initial_saved_jobs["status"])
            selected_saved_job_id_state = gr.State(value=None)
            selected_saved_job_match_state = gr.State(value=None)
            saved_job_edit_mode_state = gr.State(value=False)
            saved_job_delete_confirm_state = gr.State(value=False)

            with gr.Tabs(selected="job-search"):
                with gr.Tab("Job Search", id="job-search"):
                    gr.Markdown(
                        '<p class="tab-copy">Move from resume selection to analysis, then refine the search and review the best-matching roles.</p>'
                    )
                    source_type = gr.Radio(
                        label="Resume source",
                        choices=[("PDF", "pdf"), ("Reactive Resume", "rxresume")],
                        value=initial_source,
                    )
                    rxresume_options_state = gr.State(value=[])
                    candidate_profile_state = gr.State(value=None)
                    analysis_token_state = gr.State(value="")
                    matches_state = gr.State(value=[])
                    visible_matches_state = gr.State(value=[])
                    search_attempted_state = gr.State(value=False)
                    generated_artifacts_state = gr.State(value={})
                    selected_result_match_state = gr.State(value=None)

                    search_status = gr.HTML(_status_html(initial_pdf_draft["status_text"]))

                    with gr.Group(elem_classes=["step-card"]):
                        step_source_header = gr.HTML(
                            _step_header_html(1, "Resume Source", "Complete" if initial_pdf_draft["saved_resume_name"] else "Current")
                        )
                        resume_source_summary = gr.HTML(
                            _resume_source_summary_html(
                                source_type="pdf",
                                saved_resume_name=initial_pdf_draft["saved_resume_name"],
                                rxresume_resume_id="",
                                rxresume_options=[],
                            )
                        )
                        with gr.Group(visible=True) as pdf_group:
                            saved_pdf_name = gr.Dropdown(
                                label="Saved PDF resumes",
                                choices=controller.saved_resume_choices(),
                                value=controller.default_saved_resume_name(),
                            )
                            uploaded_pdf = gr.File(label="Upload PDF resume", file_types=[".pdf"], type="filepath")
                        with gr.Group(visible=False) as rxresume_group:
                            load_resumes_button = gr.Button("Load resumes", variant="secondary")
                            rxresume_resume_id = gr.Dropdown(label="Reactive Resume entry", choices=[], value=None)

                    with gr.Group(elem_classes=["step-card"]):
                        step_analyze_header = gr.HTML(
                            _step_header_html(2, "Analyze Resume", "Current" if initial_pdf_draft["saved_resume_name"] else "Ready")
                        )
                        candidate_summary = gr.HTML(_candidate_summary_html(None, location_used=""))
                        analyze_resume_button = gr.Button("Analyze resume", variant="primary")

                    with gr.Group(elem_classes=["step-card"]):
                        step_preferences_header = gr.HTML(_step_header_html(3, "Search Preferences", "Ready"))
                        with gr.Row():
                            location_override = gr.Textbox(
                                label="Location",
                                placeholder="Leave blank to use the analyzed resume location",
                            )
                            include_remote = gr.Checkbox(label="Include remote jobs", value=True)
                        search_terms_text = gr.Textbox(
                            label="Search terms",
                            lines=3,
                            placeholder="One search term per line, for example:\nMachine Learning Engineer\nApplied Scientist",
                        )
                        with gr.Row():
                            results_sort_by = gr.Dropdown(label="Sort by", choices=RESULT_SORT_CHOICES, value="Best match")
                            results_filter_text = gr.Textbox(
                                label="Filter title or company",
                                placeholder="Type to narrow the current result list",
                            )
                        find_jobs_button = gr.Button("Find jobs", variant="primary", interactive=False)

                    with gr.Group(elem_classes=["step-card"]):
                        step_results_header = gr.HTML(_step_header_html(4, "Results", "Ready"))
                        results_table = gr.Dataframe(
                            value=_empty_results_frame(),
                            headers=RESULTS_TABLE_HEADERS,
                            interactive=False,
                            datatype=["number", "str", "str", "str", "str", "str", "html", "str"],
                            wrap=False,
                            row_count=10,
                            max_height=620,
                            show_fullscreen_button=True,
                            column_widths=[90, 220, 220, 190, 170, 180, 320, 100],
                            elem_id="job-results-table",
                        )
                        with gr.Group(visible=False, elem_classes=["detail-card"]) as selected_job_group:
                            selected_job_detail = gr.HTML()
                            with gr.Row(visible=False) as selected_job_actions_row:
                                save_job_button = gr.Button("Save job", variant="secondary", interactive=False)
                                create_custom_resume_button = gr.Button(
                                    "Create custom resume and cover letter",
                                    variant="secondary",
                                    interactive=False,
                                )
                            with gr.Row():
                                resume_pdf_download = gr.File(label="Tailored resume PDF", visible=False)
                                cover_letter_download = gr.File(label="Cover letter", visible=False)

                with gr.Tab(_saved_jobs_tab_label(initial_saved_jobs["count"]), id="saved-jobs") as saved_jobs_tab:
                    gr.Markdown(
                        '<p class="tab-copy">Browse saved roles as a list first, then open a summary or edit the details only when needed.</p>'
                    )
                    saved_jobs_status = gr.HTML(_status_html(initial_saved_jobs["status"]))
                    saved_jobs_table = gr.Dataframe(
                        value=saved_jobs_frame_from_state(initial_saved_jobs["saved_jobs_state"]),
                        headers=SAVED_JOBS_TABLE_HEADERS,
                        interactive=False,
                        datatype=["number", "str", "str", "str", "str", "str", "html", "str"],
                        wrap=False,
                        row_count=10,
                        max_height=620,
                        show_fullscreen_button=True,
                        column_widths=[90, 220, 220, 190, 170, 180, 320, 180],
                        elem_id="saved-jobs-table",
                    )
                    with gr.Group(visible=False, elem_classes=["detail-card"]) as saved_job_summary_group:
                        saved_job_summary = gr.HTML()
                        saved_job_apply_link = gr.HTML()
                        with gr.Row(visible=False) as saved_job_actions_row:
                            edit_saved_job_button = gr.Button("Edit", variant="secondary", interactive=False)
                            delete_saved_job_button = gr.Button("Delete", variant="stop", interactive=False)
                            cancel_delete_saved_job_button = gr.Button("Cancel", variant="secondary", visible=False)
                            create_saved_job_resume_button = gr.Button(
                                "Create custom resume and cover letter",
                                variant="secondary",
                                interactive=False,
                            )
                    with gr.Group(visible=False, elem_classes=["detail-card"]) as saved_job_edit_group:
                        with gr.Row():
                            saved_job_title = gr.Textbox(label="Title")
                            saved_job_company = gr.Textbox(label="Company")
                        with gr.Row():
                            saved_job_location = gr.Textbox(label="Location")
                            saved_job_pay_range = gr.Textbox(label="Pay Range")
                        with gr.Row():
                            saved_job_via = gr.Textbox(label="Source")
                            saved_job_posted_at = gr.Textbox(label="Posted")
                            saved_job_score = gr.Number(label="Score", precision=0)
                        with gr.Row():
                            saved_job_apply_url = gr.Textbox(label="Apply URL")
                            saved_job_remote_flag = gr.Checkbox(label="Remote job", value=False)
                        with gr.Row():
                            saved_job_matched_skills = gr.Textbox(label="Matched skills", lines=3)
                            saved_job_missing_signals = gr.Textbox(label="Missing signals", lines=3)
                        saved_job_description = gr.Textbox(label="Description", lines=6)
                        saved_job_rationale = gr.Textbox(label="Rationale", lines=4)
                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                saved_job_provider = gr.Textbox(label="Provider")
                                saved_job_provider_job_id = gr.Textbox(label="Provider job ID")
                            saved_job_share_url = gr.Textbox(label="Share URL")
                        with gr.Row():
                            save_saved_job_changes_button = gr.Button("Save changes", variant="primary")
                            cancel_saved_job_edit_button = gr.Button("Cancel", variant="secondary")
                    with gr.Row():
                        saved_resume_pdf_download = gr.File(label="Tailored resume PDF", visible=False)
                        saved_cover_letter_download = gr.File(label="Cover letter", visible=False)

        search_view_outputs = [
            pdf_group,
            rxresume_group,
            saved_pdf_name,
            rxresume_resume_id,
            rxresume_options_state,
            resume_source_summary,
            candidate_summary,
            search_status,
            search_status_text_state,
            step_source_header,
            step_analyze_header,
            step_preferences_header,
            step_results_header,
            location_override,
            include_remote,
            search_terms_text,
            results_sort_by,
            results_filter_text,
            candidate_profile_state,
            analysis_token_state,
            find_jobs_button,
            matches_state,
            visible_matches_state,
            search_attempted_state,
            generated_artifacts_state,
            results_table,
            selected_result_match_state,
            selected_job_group,
            selected_job_detail,
            selected_job_actions_row,
            save_job_button,
            create_custom_resume_button,
            resume_pdf_download,
            cover_letter_download,
        ]

        saved_jobs_view_outputs = [
            saved_jobs_tab,
            saved_jobs_state,
            saved_jobs_table,
            saved_jobs_status,
            saved_jobs_status_text_state,
            selected_saved_job_id_state,
            selected_saved_job_match_state,
            saved_job_summary_group,
            saved_job_summary,
            saved_job_apply_link,
            saved_job_actions_row,
            edit_saved_job_button,
            delete_saved_job_button,
            cancel_delete_saved_job_button,
            create_saved_job_resume_button,
            saved_job_edit_group,
            saved_job_provider,
            saved_job_provider_job_id,
            saved_job_title,
            saved_job_company,
            saved_job_location,
            saved_job_pay_range,
            saved_job_via,
            saved_job_posted_at,
            saved_job_apply_url,
            saved_job_share_url,
            saved_job_remote_flag,
            saved_job_score,
            saved_job_matched_skills,
            saved_job_missing_signals,
            saved_job_description,
            saved_job_rationale,
            saved_resume_pdf_download,
            saved_cover_letter_download,
            saved_job_edit_mode_state,
            saved_job_delete_confirm_state,
        ]

        setup_source_type.change(toggle_setup_source, inputs=[setup_source_type], outputs=[setup_pdf_group], queue=False)
        for component in (setup_openai_api_key, setup_serpapi_api_key, setup_rxresume_api_key, setup_source_type, setup_pdf_file):
            component.change(
                validate_setup_inputs,
                inputs=[
                    setup_openai_api_key,
                    setup_serpapi_api_key,
                    setup_rxresume_api_key,
                    setup_source_type,
                    setup_pdf_file,
                ],
                outputs=[save_setup_button, setup_status],
                queue=False,
            )

        save_setup_button.click(
            save_setup_ui,
            inputs=[
                setup_openai_api_key,
                setup_openai_model,
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
                active_source_state,
                source_drafts_state,
                settings_openai_api_key,
                settings_openai_model,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
                *search_view_outputs,
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
                settings_openai_model,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
            ],
            outputs=[
                settings_status,
                settings_openai_api_key,
                settings_openai_model,
                settings_serpapi_api_key,
                settings_rxresume_api_key,
                settings_rxresume_api_url,
            ],
            queue=False,
        )

        source_type.change(
            switch_source_ui,
            inputs=[
                source_type,
                active_source_state,
                source_drafts_state,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                selected_result_match_state,
                search_status_text_state,
                saved_jobs_state,
            ],
            outputs=[active_source_state, source_drafts_state, *search_view_outputs],
            queue=False,
        )

        uploaded_pdf.change(
            store_uploaded_pdf_ui,
            inputs=[
                uploaded_pdf,
                source_type,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                include_remote,
                results_sort_by,
                results_filter_text,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        saved_pdf_name.input(
            reset_current_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                include_remote,
                results_sort_by,
                results_filter_text,
                search_status_text_state,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        rxresume_resume_id.input(
            reset_current_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                include_remote,
                results_sort_by,
                results_filter_text,
                search_status_text_state,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        load_resumes_button.click(
            load_resumes_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                settings_rxresume_api_url,
                saved_jobs_state,
                include_remote,
                results_sort_by,
                results_filter_text,
            ],
            outputs=search_view_outputs,
        )

        analyze_resume_button.click(
            preview_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                include_remote,
                results_sort_by,
                results_filter_text,
                settings_openai_model,
                settings_rxresume_api_url,
            ],
            outputs=search_view_outputs,
        )

        find_jobs_button.click(
            find_jobs_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                results_sort_by,
                results_filter_text,
                settings_openai_model,
                settings_rxresume_api_url,
            ],
            outputs=search_view_outputs,
        )

        results_sort_by.change(
            refresh_results_controls_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                search_status_text_state,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        results_filter_text.change(
            refresh_results_controls_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                search_status_text_state,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        results_table.select(
            select_job_result_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                search_status_text_state,
            ],
            outputs=search_view_outputs,
            queue=False,
        )

        save_job_button.click(
            save_selected_job_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                selected_result_match_state,
            ],
            outputs=[*search_view_outputs, *saved_jobs_view_outputs],
            queue=False,
        )

        create_custom_resume_button.click(
            create_custom_resume_ui,
            inputs=[
                source_type,
                saved_pdf_name,
                rxresume_resume_id,
                rxresume_options_state,
                saved_jobs_state,
                location_override,
                include_remote,
                search_terms_text,
                candidate_profile_state,
                analysis_token_state,
                matches_state,
                search_attempted_state,
                generated_artifacts_state,
                results_sort_by,
                results_filter_text,
                selected_result_match_state,
                settings_openai_model,
                settings_rxresume_api_url,
            ],
            outputs=search_view_outputs,
        )

        saved_jobs_tab.select(
            open_saved_jobs_tab_ui,
            inputs=[saved_jobs_state, selected_saved_job_id_state, saved_jobs_status_text_state],
            outputs=saved_jobs_view_outputs,
            queue=False,
        )

        saved_jobs_table.select(
            select_saved_job_ui,
            inputs=[saved_jobs_state],
            outputs=saved_jobs_view_outputs,
            queue=False,
        )

        edit_saved_job_button.click(
            enter_saved_job_edit_ui,
            inputs=[saved_jobs_state, selected_saved_job_id_state],
            outputs=saved_jobs_view_outputs,
            queue=False,
        )

        cancel_saved_job_edit_button.click(
            cancel_saved_job_edit_ui,
            inputs=[saved_jobs_state, selected_saved_job_id_state, saved_jobs_status_text_state],
            outputs=saved_jobs_view_outputs,
            queue=False,
        )

        delete_saved_job_button.click(
            toggle_delete_saved_job_ui,
            inputs=[
                saved_jobs_state,
                selected_saved_job_id_state,
                saved_job_delete_confirm_state,
                matches_state,
                results_sort_by,
                results_filter_text,
            ],
            outputs=[*saved_jobs_view_outputs, results_table],
            queue=False,
        )

        cancel_delete_saved_job_button.click(
            cancel_delete_saved_job_ui,
            inputs=[saved_jobs_state, selected_saved_job_id_state],
            outputs=saved_jobs_view_outputs,
            queue=False,
        )

        save_saved_job_changes_button.click(
            update_saved_job_ui,
            inputs=[
                selected_saved_job_id_state,
                saved_job_provider,
                saved_job_provider_job_id,
                saved_job_title,
                saved_job_company,
                saved_job_location,
                saved_job_pay_range,
                saved_job_via,
                saved_job_posted_at,
                saved_job_apply_url,
                saved_job_share_url,
                saved_job_remote_flag,
                saved_job_score,
                saved_job_matched_skills,
                saved_job_missing_signals,
                saved_job_description,
                saved_job_rationale,
                matches_state,
                results_sort_by,
                results_filter_text,
            ],
            outputs=[*saved_jobs_view_outputs, results_table],
            queue=False,
        )

        create_saved_job_resume_button.click(
            create_saved_job_resume_ui,
            inputs=[
                source_type,
                rxresume_resume_id,
                candidate_profile_state,
                saved_jobs_state,
                selected_saved_job_id_state,
                selected_saved_job_match_state,
                settings_openai_model,
                settings_rxresume_api_url,
            ],
            outputs=saved_jobs_view_outputs,
        )

    demo.queue(default_concurrency_limit=2)
    return demo
