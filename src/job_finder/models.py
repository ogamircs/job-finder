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


class SavedJobRecord(BaseModel):
    id: int
    match: ScoredJobMatch
    created_at: str = ""
    updated_at: str = ""
    application_status: str = ""
    application_run_id: str = ""
    application_run_path: str = ""
    last_applied_at: str = ""
    application_error: str = ""

    @field_validator(
        "created_at",
        "updated_at",
        "application_status",
        "application_run_id",
        "application_run_path",
        "last_applied_at",
        "application_error",
        mode="before",
    )
    @classmethod
    def _clean_saved_job_timestamps(cls, value: Any) -> str:
        return str(value or "").strip()


class TailoredSkillCategory(BaseModel):
    name: str
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, value: Any) -> list[str]:
        return _clean_unique_strings(value)


class TailoredApplicationContent(BaseModel):
    headline: str = ""
    summary: str = ""
    skills: list[TailoredSkillCategory] = Field(default_factory=list)
    cover_letter: str = ""

    @field_validator("headline", "summary", "cover_letter", mode="before")
    @classmethod
    def _clean_text_fields(cls, value: Any) -> str:
        return str(value or "").strip()


class GeneratedApplicationArtifacts(BaseModel):
    base_resume_id: str
    remote_resume_id: str
    company: str
    job_title: str
    generated_at: str
    artifact_dir: str
    pdf_url: str
    pdf_path: str
    cover_letter_path: str
    resume_json_path: str
    metadata_path: str

    @field_validator(
        "base_resume_id",
        "remote_resume_id",
        "company",
        "job_title",
        "generated_at",
        "artifact_dir",
        "pdf_url",
        "pdf_path",
        "cover_letter_path",
        "resume_json_path",
        "metadata_path",
        mode="before",
    )
    @classmethod
    def _clean_artifact_fields(cls, value: Any) -> str:
        return str(value or "").strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    # A bool is not a meaningful integer answer for fields like salary or notice
    # period — refuse it instead of silently becoming 0/1.
    if isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_dict(value: Any) -> dict[str, str]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        cleaned: dict[str, str] = {}
        for key, raw in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            cleaned[key_text] = str(raw or "").strip()
        return cleaned
    return {}


class ApplicantProfile(BaseModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location_city: str = ""
    location_region: str = ""
    location_country: str = ""
    postal_code: str = ""
    work_authorization: str = ""
    requires_sponsorship: bool = False
    willing_to_relocate: bool = False
    salary_expectation_min: int | None = None
    salary_expectation_max: int | None = None
    salary_currency: str = "USD"
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    years_experience_override: float | None = None
    preferred_pronouns: str = ""
    desired_start_date: str = ""
    notice_period_weeks: int | None = None
    default_cover_letter_signoff: str = ""
    extra_answers: dict[str, str] = Field(default_factory=dict)
    demographics: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "full_name",
        "email",
        "phone",
        "location_city",
        "location_region",
        "location_country",
        "postal_code",
        "work_authorization",
        "salary_currency",
        "linkedin_url",
        "github_url",
        "portfolio_url",
        "preferred_pronouns",
        "desired_start_date",
        "default_cover_letter_signoff",
        mode="before",
    )
    @classmethod
    def _clean_applicant_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("requires_sponsorship", "willing_to_relocate", mode="before")
    @classmethod
    def _clean_applicant_bool(cls, value: Any) -> bool:
        return _coerce_bool(value)

    @field_validator(
        "salary_expectation_min",
        "salary_expectation_max",
        "notice_period_weeks",
        mode="before",
    )
    @classmethod
    def _clean_applicant_optional_int(cls, value: Any) -> int | None:
        return _coerce_optional_int(value)

    @field_validator("years_experience_override", mode="before")
    @classmethod
    def _clean_applicant_optional_float(cls, value: Any) -> float | None:
        return _coerce_optional_float(value)

    @field_validator("extra_answers", "demographics", mode="before")
    @classmethod
    def _clean_applicant_dict(cls, value: Any) -> dict[str, str]:
        return _coerce_str_dict(value)
