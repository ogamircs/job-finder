from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

import httpx

from .models import JobPosting

_CANADA_REGIONS = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "MB": "Manitoba",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YT": "Yukon",
}

_US_REGIONS = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}

_PAY_PATTERN = re.compile(
    r"(\$|£|€|\bUSD\b|\bCAD\b|\bEUR\b|\bGBP\b|\d+\s?[kK]\b|"
    r"\bper\s+(hour|year|month|week)\b|"
    r"/(hour|hr|year|yr|month|mo|week|wk)\b|"
    r"\ba\s+(hour|year|month|week)\b)"
)


class JobProvider(ABC):
    @abstractmethod
    def search(self, query: str, location: str | None, remote: bool) -> list[JobPosting]:
        raise NotImplementedError


def _safe_serpapi_error(exc: Exception) -> ValueError:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        detail = (response.text or "").strip()
        if len(detail) > 240:
            detail = f"{detail[:237]}..."
        return ValueError(f"SerpApi request failed ({response.status_code}): {detail or 'No error details returned.'}")
    return ValueError("SerpApi request failed.")


def _normalize_serpapi_location(location: str | None) -> str | None:
    if location is None:
        return None

    cleaned = location.strip()
    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        return cleaned

    city, region = parts
    region_code = region.upper()
    if region_code in _CANADA_REGIONS:
        return f"{city}, {_CANADA_REGIONS[region_code]}, Canada"
    if region_code in _US_REGIONS:
        return f"{city}, {_US_REGIONS[region_code]}, United States"
    return cleaned


def _best_apply_option(options: list[dict[str, Any]]) -> str:
    if not options:
        return ""

    def score(option: dict[str, Any]) -> tuple[int, str]:
        title = str(option.get("title") or "").lower()
        link = str(option.get("link") or "").strip()
        title_score = 0
        if "company" in title or "employer" in title or "career" in title:
            title_score += 2
        if "linkedin" in title or "glassdoor" in title or "indeed" in title:
            title_score -= 1
        return (title_score, link)

    ranked = sorted(options, key=score, reverse=True)
    for option in ranked:
        link = str(option.get("link") or "").strip()
        if link:
            return link
    return ""


def _first_related_link(raw_job: dict[str, Any]) -> str:
    related_links = raw_job.get("related_links") or []
    if not isinstance(related_links, list):
        return ""
    for item in related_links:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link") or "").strip()
        if link:
            return link
    return ""


def _iter_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_iter_strings(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for nested in value:
            values.extend(_iter_strings(nested))
        return values
    return []


def _extract_pay_range(raw_job: dict[str, Any]) -> str:
    detected_extensions = raw_job.get("detected_extensions") or {}
    candidates: list[str] = []
    for value in (
        raw_job.get("salary"),
        raw_job.get("salary_highlights"),
        detected_extensions.get("salary"),
        detected_extensions.get("base_salary"),
        detected_extensions,
        raw_job.get("extensions"),
    ):
        candidates.extend(_iter_strings(value))

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        if any(char.isdigit() for char in candidate) and _PAY_PATTERN.search(candidate):
            return candidate
    return ""


def parse_serpapi_job(raw_job: dict[str, Any]) -> JobPosting:
    apply_options = raw_job.get("apply_options") or []
    detected_extensions = raw_job.get("detected_extensions") or {}
    location = str(raw_job.get("location") or "").strip()
    schedule_type = str(detected_extensions.get("schedule_type") or "").strip()
    apply_url = _best_apply_option(apply_options)
    if not apply_url:
        apply_url = _first_related_link(raw_job)

    return JobPosting(
        provider="serpapi_google_jobs",
        provider_job_id=str(raw_job.get("job_id") or "").strip(),
        title=str(raw_job.get("title") or "").strip(),
        company=str(raw_job.get("company_name") or "").strip(),
        location=location,
        pay_range=_extract_pay_range(raw_job),
        via=str(raw_job.get("via") or "").strip(),
        description=str(raw_job.get("description") or "").strip(),
        posted_at=str(detected_extensions.get("posted_at") or "").strip(),
        remote_flag="remote" in location.casefold() or "remote" in schedule_type.casefold(),
        apply_url=apply_url,
        share_url=str(raw_job.get("share_link") or apply_url).strip(),
    )


class SerpApiGoogleJobsProvider(JobProvider):
    endpoint = "https://serpapi.com/search.json"

    def __init__(self, *, api_key: str, http_client: Any | None = None) -> None:
        self.api_key = api_key
        self.http_client = http_client

    def search(self, query: str, location: str | None, remote: bool) -> list[JobPosting]:
        params = {
            "engine": "google_jobs",
            "q": self._build_query(query, remote),
            "api_key": self.api_key,
        }
        normalized_location = _normalize_serpapi_location(location)
        if normalized_location:
            params["location"] = normalized_location

        try:
            if self.http_client is not None:
                response = self.http_client.get(self.endpoint, params=params, timeout=30.0)
                response.raise_for_status()
                payload = response.json()
            else:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(self.endpoint, params=params)
                    response.raise_for_status()
                    payload = response.json()
        except Exception as exc:
            raise _safe_serpapi_error(exc) from exc

        jobs = payload.get("jobs_results") or payload.get("jobs") or []
        parsed_jobs: list[JobPosting] = []
        for raw_job in jobs:
            job = parse_serpapi_job(raw_job)
            if job.title and job.company and job.apply_url:
                parsed_jobs.append(job)
        return parsed_jobs

    @staticmethod
    def _build_query(query: str, remote: bool) -> str:
        cleaned = query.strip()
        if remote and "remote" not in cleaned.casefold():
            return f"{cleaned} remote"
        return cleaned
