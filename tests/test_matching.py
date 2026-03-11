from job_finder.matching import (
    build_search_queries,
    dedupe_jobs,
    finalize_matches,
    find_job_matches,
    score_jobs,
    prefilter_jobs,
)
from job_finder.models import CandidateProfile, JobPosting, ScoredJobMatch, SearchRequest


def make_job(**overrides):
    return JobPosting(
        provider="serpapi_google_jobs",
        provider_job_id=overrides.get("provider_job_id", "job-1"),
        title=overrides.get("title", "Machine Learning Engineer"),
        company=overrides.get("company", "Acme AI"),
        location=overrides.get("location", "Toronto, ON"),
        via=overrides.get("via", "via LinkedIn"),
        description=overrides.get(
            "description",
            "Python, machine learning, and recommendation systems.",
        ),
        posted_at=overrides.get("posted_at", "2 days ago"),
        remote_flag=overrides.get("remote_flag", False),
        apply_url=overrides.get("apply_url", "https://example.com/jobs/1"),
        share_url=overrides.get("share_url", "https://google.com/jobs/1"),
    )


def test_build_search_queries_returns_max_three_titles():
    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=[
            "Senior Machine Learning Engineer",
            "Applied Scientist",
            "AI Product Engineer",
            "Ignored Title",
        ],
        years_experience=8,
        top_skills=["Python", "OpenAI", "Ranking"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )

    assert build_search_queries(profile) == [
        "Senior Machine Learning Engineer",
        "Applied Scientist",
        "AI Product Engineer",
    ]


def test_build_search_queries_prefers_user_edited_search_terms():
    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "OpenAI", "Ranking"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )

    assert build_search_queries(profile, ["Applied Scientist", "ML Platform Engineer"]) == [
        "Applied Scientist",
        "ML Platform Engineer",
    ]


def test_dedupe_jobs_uses_provider_id_and_fallback_signature():
    jobs = [
        make_job(provider_job_id="job-1", apply_url="https://example.com/jobs/1"),
        make_job(provider_job_id="job-1", apply_url="https://example.com/jobs/1?duplicate=true"),
        make_job(
            provider_job_id="",
            title="Backend Engineer",
            company="Example",
            apply_url="https://example.com/jobs/backend",
            share_url="https://google.com/jobs/backend/1",
        ),
        make_job(
            provider_job_id="",
            title="Backend Engineer",
            company="Example",
            apply_url="https://example.com/jobs/backend",
            share_url="https://google.com/jobs/backend/2",
        ),
    ]

    deduped = dedupe_jobs(jobs)

    assert len(deduped) == 2


def test_prefilter_jobs_prioritizes_title_and_skill_overlap():
    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "LLM", "Ranking"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )
    jobs = [
        make_job(
            provider_job_id="best",
            title="Machine Learning Engineer",
            description="Python ranking and LLM systems.",
        ),
        make_job(
            provider_job_id="middle",
            title="Software Engineer",
            description="Python services and APIs.",
            apply_url="https://example.com/jobs/2",
        ),
        make_job(
            provider_job_id="weak",
            title="Account Executive",
            description="Sales pipeline ownership.",
            apply_url="https://example.com/jobs/3",
        ),
    ]

    filtered = prefilter_jobs(profile, jobs, limit=2)

    assert [job.provider_job_id for job in filtered] == ["best", "middle"]


def test_finalize_matches_applies_threshold_and_limit():
    matches = [
        ScoredJobMatch(
            job=make_job(provider_job_id="9"),
            score_10=9,
            rationale="Strong fit",
            matched_skills=["Python"],
            missing_signals=[],
        ),
        ScoredJobMatch(
            job=make_job(provider_job_id="8", apply_url="https://example.com/jobs/8"),
            score_10=8,
            rationale="Good fit",
            matched_skills=["Python"],
            missing_signals=[],
        ),
        ScoredJobMatch(
            job=make_job(provider_job_id="6", apply_url="https://example.com/jobs/6"),
            score_10=6,
            rationale="Weak fit",
            matched_skills=[],
            missing_signals=["LLM"],
        ),
    ]

    kept = finalize_matches(matches, threshold=7, limit=1)

    assert [match.job.provider_job_id for match in kept] == ["9"]


def test_find_job_matches_searches_remote_without_location_filter(monkeypatch):
    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "OpenAI"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )

    class FakeProvider:
        def __init__(self):
            self.calls = []

        def search(self, query, location, remote):
            self.calls.append((query, location, remote))
            return [make_job(provider_job_id=f"{query}-{remote}-{location or 'none'}")]

    monkeypatch.setattr(
        "job_finder.matching.score_jobs",
        lambda *args, **kwargs: [
            ScoredJobMatch(
                job=make_job(provider_job_id="job-1"),
                score_10=9,
                rationale="Strong fit",
                matched_skills=["Python"],
                missing_signals=[],
            )
        ],
    )

    provider = FakeProvider()
    matches = find_job_matches(
        profile,
        request=SearchRequest(
            source_type="pdf",
            location_override="Toronto, ON",
            include_remote=True,
        ),
        provider=provider,
        openai_key="sk-test",
    )

    assert matches[0].score_10 == 9
    assert provider.calls == [
        ("Machine Learning Engineer", "Toronto, ON", False),
        ("Machine Learning Engineer", None, True),
    ]


def test_find_job_matches_uses_search_term_override(monkeypatch):
    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "OpenAI"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )

    class FakeProvider:
        def __init__(self):
            self.calls = []

        def search(self, query, location, remote):
            self.calls.append((query, location, remote))
            return [make_job(provider_job_id=f"{query}-{remote}")]

    monkeypatch.setattr(
        "job_finder.matching.score_jobs",
        lambda *args, **kwargs: [
            ScoredJobMatch(
                job=make_job(provider_job_id="job-1"),
                score_10=9,
                rationale="Strong fit",
                matched_skills=["Python"],
                missing_signals=[],
            )
        ],
    )

    provider = FakeProvider()
    find_job_matches(
        profile,
        request=SearchRequest(
            source_type="pdf",
            location_override="Toronto, ON",
            include_remote=False,
            search_terms=["Applied Scientist", "ML Platform Engineer"],
        ),
        provider=provider,
        openai_key="sk-test",
    )

    assert provider.calls == [
        ("Applied Scientist", "Toronto, ON", False),
        ("ML Platform Engineer", "Toronto, ON", False),
    ]


def test_score_jobs_uses_supported_verbosity_for_gpt4o():
    class FakeResponsesClient:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return type(
                "FakeResponse",
                (),
                {
                    "output_text": '{"matches":[{"job_id":"job-1","score_10":9,"rationale":"Strong fit","matched_skills":["Python"],"missing_signals":[]}]}'
                },
            )()

    class FakeOpenAIClient:
        def __init__(self):
            self.responses = FakeResponsesClient()

    profile = CandidateProfile(
        name="Ada Lovelace",
        headline="Applied AI Engineer",
        inferred_location="Toronto, ON",
        target_titles=["Machine Learning Engineer"],
        years_experience=8,
        top_skills=["Python", "OpenAI"],
        industries=["Software"],
        summary_for_matching="Experienced AI engineer.",
    )
    client = FakeOpenAIClient()

    score_jobs(
        profile,
        [make_job(provider_job_id="job-1")],
        openai_key="sk-test",
        client=client,
    )

    assert client.responses.calls[0]["text"]["verbosity"] == "medium"
