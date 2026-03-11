from __future__ import annotations

import base64
import html
import json
import re
from datetime import date, datetime
from typing import Any

import httpx
from openai import OpenAI

from .models import CandidateProfile, ResumeOption

DEFAULT_RXRESUME_RESUMES_URL = "https://rxresu.me/api/openapi/resumes"


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def build_rxresume_resumes_url(base_url: str) -> str:
    cleaned = _normalize_base_url(base_url)
    if cleaned.endswith("/api/openapi/resumes") or cleaned.endswith("/api/resume"):
        return cleaned
    return f"{cleaned}/api/openapi/resumes"


def build_rxresume_resume_url(base_url: str, resume_id: str) -> str:
    return f"{build_rxresume_resumes_url(base_url)}/{resume_id.strip()}"


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("items"), list):
        return payload["data"]["items"]
    return []


def _request_json(
    url: str,
    *,
    api_key: str,
    http_client: Any | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Accept": "application/json",
    }
    if http_client is not None:
        response = http_client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def candidate_profile_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "headline": {"type": "string"},
            "inferred_location": {"type": "string"},
            "target_titles": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "years_experience": {"type": "number"},
            "top_skills": {
                "type": "array",
                "items": {"type": "string"},
            },
            "industries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "summary_for_matching": {"type": "string"},
        },
        "required": [
            "name",
            "headline",
            "inferred_location",
            "target_titles",
            "years_experience",
            "top_skills",
            "industries",
            "summary_for_matching",
        ],
    }


def list_rxresume_resumes(
    base_url: str,
    api_key: str,
    *,
    http_client: Any | None = None,
) -> list[ResumeOption]:
    payload = _request_json(
        build_rxresume_resumes_url(base_url),
        api_key=api_key,
        http_client=http_client,
    )
    options: list[ResumeOption] = []
    for item in _extract_items(payload):
        resume_id = str(item.get("id") or item.get("slug") or "").strip()
        label = (
            str(item.get("title") or item.get("name") or item.get("label") or resume_id).strip()
        )
        if resume_id and label:
            options.append(ResumeOption(id=resume_id, label=label))
    return options


def load_candidate_profile_from_rxresume(
    base_url: str,
    api_key: str,
    resume_id: str,
    *,
    http_client: Any | None = None,
) -> CandidateProfile:
    payload = _request_json(
        build_rxresume_resume_url(base_url, resume_id),
        api_key=api_key,
        http_client=http_client,
    )
    resume = payload.get("item") or payload.get("data") or payload
    return normalize_rxresume_resume(resume)


def _coerce_location(location_value: Any) -> str:
    if isinstance(location_value, str):
        return location_value.strip()
    if isinstance(location_value, dict):
        parts = [
            str(location_value.get(key) or "").strip()
            for key in ("city", "region", "postalCode", "country")
        ]
        return ", ".join(part for part in parts if part)
    return ""


def _extract_section_items(payload: dict[str, Any], section_name: str) -> list[dict[str, Any]]:
    sections = payload.get("sections") or {}
    section = sections.get(section_name)
    if isinstance(section, dict) and isinstance(section.get("items"), list):
        return section["items"]
    legacy = payload.get(section_name)
    if isinstance(legacy, dict) and isinstance(legacy.get("items"), list):
        return legacy["items"]
    if isinstance(legacy, list):
        return legacy
    return []


def _html_to_text(value: Any) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|ul|ol|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _parse_partial_date(value: str | None) -> date | None:
    if not value:
        return None

    cleaned = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%b %Y", "%B %Y"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        if fmt == "%Y":
            return date(parsed.year, 1, 1)
        if fmt == "%Y-%m":
            return date(parsed.year, parsed.month, 1)
        if fmt in ("%b %Y", "%B %Y"):
            return date(parsed.year, parsed.month, 1)
        return parsed.date()
    return None


def _parse_period_dates(value: Any) -> tuple[date | None, date | None]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return (None, None)

    for separator in (" - ", " – ", " — "):
        if separator not in cleaned:
            continue
        start_text, end_text = cleaned.split(separator, 1)
        start = _parse_partial_date(start_text)
        end_value = end_text.strip().casefold()
        if end_value in {"present", "current", "now"}:
            end = date.today()
        else:
            end = _parse_partial_date(end_text)
        return (start, end)

    return (_parse_partial_date(cleaned), None)


def _months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def _compute_years_experience(experiences: list[dict[str, Any]]) -> float:
    months = 0
    today = date.today()
    for item in experiences:
        start = _parse_partial_date(item.get("startDate") or item.get("date"))
        end = _parse_partial_date(item.get("endDate"))
        if start is None:
            start, end = _parse_period_dates(item.get("period"))
        end = end or today
        if start is None:
            continue
        months += _months_between(start, end)
    return round(months / 12, 1)


def _dedupe(values: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _flatten_experience_items(experiences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in experiences:
        roles = item.get("roles") or []
        if isinstance(roles, list) and roles:
            for role in roles:
                if not isinstance(role, dict):
                    continue
                flattened.append(
                    {
                        **role,
                        "company": item.get("company"),
                        "location": item.get("location"),
                    }
                )
        else:
            flattened.append(item)
    return flattened


def normalize_rxresume_resume(payload: dict[str, Any]) -> CandidateProfile:
    basics = payload.get("basics") or {}
    experiences = _extract_section_items(payload, "experience")
    normalized_experiences = _flatten_experience_items(experiences)
    skill_items = _extract_section_items(payload, "skills")

    titles = [
        str(item.get("position") or item.get("title") or "").strip()
        for item in normalized_experiences
    ]
    headline = str(basics.get("label") or basics.get("headline") or "").strip()
    if headline:
        titles.append(headline)

    top_skills: list[str] = []
    for item in skill_items:
        top_skills.append(str(item.get("name") or item.get("label") or "").strip())
        keywords = item.get("keywords") or []
        if isinstance(keywords, list):
            top_skills.extend(str(keyword).strip() for keyword in keywords)

    experience_summaries = [
        _html_to_text(item.get("summary") or item.get("description"))
        for item in normalized_experiences
        if _html_to_text(item.get("summary") or item.get("description"))
    ]
    summary_section = payload.get("summary") or {}
    summary_text = basics.get("summary") or summary_section.get("content") or summary_section.get("summary") or ""
    summary_parts = [
        _html_to_text(summary_text),
        *experience_summaries[:2],
    ]

    return CandidateProfile(
        name=str(basics.get("name") or "").strip(),
        headline=headline,
        inferred_location=_coerce_location(basics.get("location")),
        target_titles=_dedupe(titles, limit=3),
        years_experience=_compute_years_experience(normalized_experiences),
        top_skills=_dedupe(top_skills),
        industries=[],
        summary_for_matching=" ".join(part for part in summary_parts if part),
    )


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return ""


def ensure_candidate_profile_has_signal(
    profile: CandidateProfile,
    *,
    error_message: str = "Could not extract enough resume details from the profile.",
) -> CandidateProfile:
    has_signal = any(
        [
            profile.name,
            profile.headline,
            profile.target_titles,
            profile.top_skills,
            profile.summary_for_matching,
        ]
    )
    if not has_signal:
        raise ValueError(error_message)
    return profile


def load_candidate_profile_from_pdf(
    pdf_bytes: bytes,
    openai_key: str,
    *,
    filename: str = "resume.pdf",
    model: str = "gpt-4o",
    client: Any | None = None,
) -> CandidateProfile:
    file_data = base64.b64encode(pdf_bytes).decode("utf-8")
    openai_client = client or OpenAI(api_key=openai_key)
    response = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": f"data:application/pdf;base64,{file_data}",
                    },
                    {
                        "type": "input_text",
                        "text": (
                            "Extract a structured candidate profile from this resume. "
                            "Prefer concrete titles and skills present in the document."
                        ),
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "candidate_profile",
                "schema": candidate_profile_response_schema(),
                "strict": True,
            },
            "verbosity": "medium",
        },
        max_output_tokens=1200,
    )
    payload = json.loads(_response_text(response))
    profile = CandidateProfile.model_validate(payload)
    return ensure_candidate_profile_has_signal(
        profile,
        error_message="Could not extract enough resume details from the PDF.",
    )
