from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .job_provider import JobProvider
from .models import CandidateProfile, JobPosting, ScoredJobMatch, SearchRequest


def build_search_queries(profile: CandidateProfile, search_terms: list[str] | None = None) -> list[str]:
    provided_terms = [term for term in (search_terms or []) if str(term).strip()]
    if provided_terms:
        return provided_terms[:3]
    titles = [title for title in profile.target_titles if title.strip()]
    if titles:
        return titles[:3]
    if profile.headline:
        return [profile.headline]
    return ["Software Engineer"]


def dedupe_jobs(jobs: list[JobPosting]) -> list[JobPosting]:
    seen: set[str] = set()
    unique: list[JobPosting] = []
    for job in jobs:
        if job.provider_job_id:
            key = f"id:{job.provider}:{job.provider_job_id}"
        else:
            key = "sig:{title}|{company}|{apply_url}".format(
                title=job.title.casefold(),
                company=job.company.casefold(),
                apply_url=job.apply_url.casefold(),
            )
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def _job_prefilter_score(profile: CandidateProfile, job: JobPosting) -> int:
    haystack = " ".join([job.title, job.description]).casefold()
    score = 0
    for title in profile.target_titles:
        title_lower = title.casefold()
        if title_lower and title_lower in job.title.casefold():
            score += 10
        elif title_lower and title_lower in haystack:
            score += 6
    for skill in profile.top_skills:
        skill_lower = skill.casefold()
        if skill_lower and skill_lower in haystack:
            score += 2
    if job.remote_flag:
        score += 1
    return score


def prefilter_jobs(
    profile: CandidateProfile,
    jobs: list[JobPosting],
    *,
    limit: int = 25,
) -> list[JobPosting]:
    ranked = sorted(
        jobs,
        key=lambda job: (-_job_prefilter_score(profile, job), job.title.casefold(), job.company.casefold()),
    )
    return ranked[:limit]


def finalize_matches(
    matches: list[ScoredJobMatch],
    *,
    threshold: int,
    limit: int,
) -> list[ScoredJobMatch]:
    filtered = [match for match in matches if match.score_10 >= threshold]
    filtered.sort(key=lambda match: (-match.score_10, match.job.title.casefold(), match.job.company.casefold()))
    return filtered[:limit]


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return ""


def _job_batch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "job_id": {"type": "string"},
                        "score_10": {"type": "integer", "minimum": 0, "maximum": 10},
                        "rationale": {"type": "string"},
                        "matched_skills": {"type": "array", "items": {"type": "string"}},
                        "missing_signals": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "job_id",
                        "score_10",
                        "rationale",
                        "matched_skills",
                        "missing_signals",
                    ],
                },
            }
        },
        "required": ["matches"],
    }


def score_jobs(
    profile: CandidateProfile,
    jobs: list[JobPosting],
    *,
    openai_key: str,
    model: str = "gpt-4o",
    client: Any | None = None,
    batch_size: int = 5,
) -> list[ScoredJobMatch]:
    if not jobs:
        return []

    openai_client = client or OpenAI(api_key=openai_key)
    scored_matches: list[ScoredJobMatch] = []

    for start_index in range(0, len(jobs), batch_size):
        batch = jobs[start_index : start_index + batch_size]
        job_map = {
            (job.provider_job_id or f"job-{start_index + offset}"): job
            for offset, job in enumerate(batch)
        }
        jobs_payload = [
            {
                "job_id": job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "remote_flag": job.remote_flag,
            }
            for job_id, job in job_map.items()
        ]
        prompt = {
            "candidate_profile": profile.model_dump(),
            "jobs": jobs_payload,
            "instructions": (
                "Score each job from 0 to 10 for resume fit. "
                "Return only concise evidence grounded in the resume and job description."
            ),
        }
        response = openai_client.responses.create(
            model=model,
            input=json.dumps(prompt),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_match_scores",
                    "schema": _job_batch_schema(),
                    "strict": True,
                },
                "verbosity": "medium",
            },
            max_output_tokens=1800,
        )
        payload = json.loads(_response_text(response))
        for item in payload.get("matches", []):
            job = job_map.get(str(item.get("job_id") or "").strip())
            if job is None:
                continue
            scored_matches.append(
                ScoredJobMatch(
                    job=job,
                    score_10=int(item.get("score_10", 0)),
                    rationale=str(item.get("rationale") or "").strip(),
                    matched_skills=item.get("matched_skills") or [],
                    missing_signals=item.get("missing_signals") or [],
                )
            )
    return scored_matches


def find_job_matches(
    profile: CandidateProfile,
    request: SearchRequest,
    provider: JobProvider,
    *,
    openai_key: str,
    openai_client: Any | None = None,
    model: str = "gpt-4o",
) -> list[ScoredJobMatch]:
    queries = build_search_queries(profile, request.search_terms)
    location = request.location_override.strip() or None
    jobs: list[JobPosting] = []

    for query in queries:
        if location:
            jobs.extend(provider.search(query, location, remote=False))
        if request.include_remote:
            jobs.extend(provider.search(query, None, remote=True))

    deduped_jobs = dedupe_jobs(jobs)
    shortlisted_jobs = prefilter_jobs(profile, deduped_jobs, limit=25)
    scored_matches = score_jobs(
        profile,
        shortlisted_jobs,
        openai_key=openai_key,
        model=model,
        client=openai_client,
    )
    return finalize_matches(
        scored_matches,
        threshold=request.threshold,
        limit=request.result_limit,
    )
