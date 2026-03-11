import httpx

from job_finder.job_provider import SerpApiGoogleJobsProvider, parse_serpapi_job


class FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeHttpResponse(self.payload)


class ErrorHttpClient:
    def get(self, url, *, params=None, timeout=None):
        request = httpx.Request(
            "GET",
            "https://serpapi.com/search.json?engine=google_jobs&api_key=secret-key&location=Toronto%2C+ON",
        )
        response = httpx.Response(
            400,
            request=request,
            text='{"error":"Invalid location"}',
        )
        raise httpx.HTTPStatusError("boom", request=request, response=response)


def test_parse_serpapi_job_prefers_company_site_and_flags_remote():
    raw_job = {
        "job_id": "job-123",
        "title": "Senior ML Engineer",
        "company_name": "Acme AI",
        "location": "Toronto, ON",
        "description": "Build LLM products with Python and retrieval.",
        "via": "via LinkedIn",
        "share_link": "https://www.google.com/search?ibp=htl;jobs",
        "detected_extensions": {
            "posted_at": "3 days ago",
            "schedule_type": "Remote",
        },
        "apply_options": [
            {
                "title": "Apply on LinkedIn",
                "link": "https://linkedin.com/jobs/view/123",
            },
            {
                "title": "Apply on Company Site",
                "link": "https://careers.acme.ai/jobs/123",
            },
        ],
    }

    job = parse_serpapi_job(raw_job)

    assert job.provider == "serpapi_google_jobs"
    assert job.provider_job_id == "job-123"
    assert job.remote_flag is True
    assert job.via == "via LinkedIn"
    assert job.apply_url == "https://careers.acme.ai/jobs/123"
    assert job.share_url == "https://www.google.com/search?ibp=htl;jobs"


def test_parse_serpapi_job_extracts_pay_range_when_available():
    raw_job = {
        "job_id": "job-456",
        "title": "Applied Scientist",
        "company_name": "Example Labs",
        "location": "Toronto, ON",
        "description": "Build ranking systems.",
        "via": "via Glassdoor",
        "detected_extensions": {
            "posted_at": "1 day ago",
            "salary": "$145K-$180K a year",
        },
        "salary_highlights": [
            {
                "salary": "$145K-$180K a year",
            }
        ],
        "extensions": ["$145K-$180K a year", "Full-time"],
        "apply_options": [
            {
                "title": "Apply on Company Site",
                "link": "https://example.com/jobs/456",
            }
        ],
    }

    job = parse_serpapi_job(raw_job)

    assert job.pay_range == "$145K-$180K a year"


def test_serpapi_provider_search_sends_remote_query_and_location():
    client = FakeHttpClient({"jobs_results": []})
    provider = SerpApiGoogleJobsProvider(api_key="serp-key", http_client=client)

    provider.search("Machine Learning Engineer", "Toronto, ON", remote=True)

    call = client.calls[0]
    assert call["url"] == "https://serpapi.com/search.json"
    assert call["params"]["engine"] == "google_jobs"
    assert call["params"]["q"] == "Machine Learning Engineer remote"
    assert call["params"]["location"] == "Toronto, Ontario, Canada"
    assert call["params"]["api_key"] == "serp-key"


def test_serpapi_provider_search_sanitizes_api_key_in_errors():
    provider = SerpApiGoogleJobsProvider(api_key="secret-key", http_client=ErrorHttpClient())

    try:
        provider.search("Machine Learning Engineer", "Toronto, ON", remote=False)
    except ValueError as exc:
        message = str(exc)
        assert "secret-key" not in message
        assert "SerpApi request failed (400)" in message
        assert "Invalid location" in message
    else:
        raise AssertionError("Expected a sanitized SerpApi error")
