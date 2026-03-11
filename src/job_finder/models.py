from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _clean_unique_strings(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)

    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


class CandidateProfile(BaseModel):
    name: str = ""
    headline: str = ""
    inferred_location: str = ""
    target_titles: list[str] = Field(default_factory=list, max_length=3)
    years_experience: float = 0.0
    top_skills: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    summary_for_matching: str = ""

    @field_validator("name", "headline", "inferred_location", "summary_for_matching", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("target_titles", mode="before")
    @classmethod
    def _clean_titles(cls, value: Any) -> list[str]:
        return _clean_unique_strings(value, limit=3)

    @field_validator("top_skills", "industries", mode="before")
    @classmethod
    def _clean_lists(cls, value: Any) -> list[str]:
        return _clean_unique_strings(value)

    @field_validator("years_experience", mode="before")
    @classmethod
    def _clean_years_experience(cls, value: Any) -> float:
        if value in (None, ""):
            return 0.0
        return round(max(float(value), 0.0), 1)


class SearchRequest(BaseModel):
    source_type: str
    location_override: str = ""
    search_terms: list[str] = Field(default_factory=list)
    include_remote: bool = True
    threshold: int = 7
    result_limit: int = 15

    @field_validator("source_type", "location_override", mode="before")
    @classmethod
    def _clean_request_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("search_terms", mode="before")
    @classmethod
    def _clean_search_terms(cls, value: Any) -> list[str]:
        return _clean_unique_strings(value, limit=3)


class ResumeOption(BaseModel):
    id: str
    label: str


class JobPosting(BaseModel):
    provider: str
    provider_job_id: str = ""
    title: str
    company: str
    location: str = ""
    pay_range: str = ""
    via: str = ""
    description: str = ""
    posted_at: str = ""
    remote_flag: bool = False
    apply_url: str
    share_url: str = ""

    @field_validator(
        "provider",
        "provider_job_id",
        "title",
        "company",
        "location",
        "pay_range",
        "via",
        "description",
        "posted_at",
        "apply_url",
        "share_url",
        mode="before",
    )
    @classmethod
    def _clean_job_text(cls, value: Any) -> str:
        return str(value or "").strip()


class ScoredJobMatch(BaseModel):
    job: JobPosting
    score_10: int = Field(ge=0, le=10)
    rationale: str = ""
    matched_skills: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)

    @field_validator("rationale", mode="before")
    @classmethod
    def _clean_rationale(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("matched_skills", "missing_signals", mode="before")
    @classmethod
    def _clean_match_lists(cls, value: Any) -> list[str]:
        return _clean_unique_strings(value)


class SearchRunResult(BaseModel):
    profile: CandidateProfile
    location_used: str = ""
    matches: list[ScoredJobMatch] = Field(default_factory=list)
    status: str = ""
