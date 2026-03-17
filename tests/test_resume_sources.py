import base64
import json

from job_finder.models import ResumeOption
from job_finder.resume_sources import (
    _compute_years_experience,
    build_rxresume_resume_url,
    build_rxresume_resumes_url,
    candidate_profile_response_schema,
    delete_rxresume_resume,
    download_file,
    export_rxresume_resume_pdf,
    import_rxresume_resume,
    load_rxresume_resume_document,
    load_candidate_profile_from_rxresume,
    list_rxresume_resumes,
    load_candidate_profile_from_pdf,
    normalize_rxresume_resume,
)


class FakeHttpResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, headers=None, params=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )
        return FakeHttpResponse(self.payload)


class FakeRequestHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, bytes):
            return FakeHttpResponse(content=response)
        return FakeHttpResponse(payload=response)

    def get(self, url, *, headers=None, timeout=None):
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "headers": headers,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, bytes):
            return FakeHttpResponse(content=response)
        return FakeHttpResponse(payload=response)


class RecordingContextClient:
    def __init__(self, outer):
        self.outer = outer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, *, headers=None):
        self.outer.calls.append({"url": url, "headers": headers})
        return FakeHttpResponse(self.outer.payload)


class RecordingHttpxFactory:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.kwargs = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return RecordingContextClient(self)


class FakeResponsesClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": json.dumps(self.payload)})()


class FakeOpenAIClient:
    def __init__(self, payload):
        self.responses = FakeResponsesClient(payload)


def test_normalize_rxresume_resume_extracts_candidate_profile():
    rxresume_payload = {
        "basics": {
            "name": "Ada Lovelace",
            "label": "Machine Learning Engineer",
            "location": "Toronto, ON",
            "summary": "Builds applied AI systems for job matching products.",
        },
        "sections": {
            "experience": {
                "items": [
                    {
                        "position": "Senior Machine Learning Engineer",
                        "company": "Acme AI",
                        "startDate": "2018-01-01",
                        "endDate": "2020-01-01",
                        "summary": "Built NLP systems with Python and vector search.",
                    },
                    {
                        "position": "Applied Scientist",
                        "company": "Example Labs",
                        "startDate": "2020-02-01",
                        "endDate": "2022-02-01",
                        "summary": "Shipped recommendation models for hiring teams.",
                    },
                ]
            },
            "skills": {
                "items": [
                    {"name": "Python", "keywords": ["FastAPI", "Pydantic"]},
                    {"name": "OpenAI", "keywords": ["GPT-4o", "structured outputs"]},
                ]
            },
        },
    }

    profile = normalize_rxresume_resume(rxresume_payload)

    assert profile.name == "Ada Lovelace"
    assert profile.headline == "Machine Learning Engineer"
    assert profile.inferred_location == "Toronto, ON"
    assert profile.target_titles[:2] == [
        "Senior Machine Learning Engineer",
        "Applied Scientist",
    ]
    assert profile.years_experience == 4.0
    assert "Python" in profile.top_skills
    assert "FastAPI" in profile.top_skills
    assert "Built NLP systems" in profile.summary_for_matching


def test_normalize_rxresume_resume_supports_hosted_openapi_payload_shape():
    rxresume_payload = {
        "basics": {
            "name": "Amir Tavasoli",
            "headline": "Director-level ML Engineering & Infrastructure Leader",
            "location": "Toronto, ON",
        },
        "summary": {
            "content": (
                "<p>Director-level ML Engineering &amp; Infrastructure leader with 10+ years "
                "building and scaling production ML platforms.</p>"
            )
        },
        "sections": {
            "experience": {
                "items": [
                    {
                        "company": "Sobeys",
                        "position": "",
                        "period": "Jul 2021 - Jan 2024",
                        "description": "",
                        "roles": [
                            {
                                "position": "Director of Data Science, Marketing",
                                "period": "Jan 2024 - Jan 2025",
                                "description": "<ul><li>Led applied AI platform strategy.</li></ul>",
                            },
                            {
                                "position": "Data Science Manager, Personalization",
                                "period": "Jul 2021 - Jan 2024",
                                "description": "<ul><li>Rebuilt personalization ML systems.</li></ul>",
                            },
                        ],
                    },
                    {
                        "company": "Home Depot",
                        "position": "Senior Data Scientist, eCommerce",
                        "period": "Jul 2019 - Jul 2021",
                        "description": "<ul><li>Built search ranking models.</li></ul>",
                        "roles": [],
                    },
                ]
            },
            "skills": {
                "items": [
                    {"name": "Languages/Data", "keywords": ["Python", "SQL"]},
                    {"name": "ML/AI", "keywords": ["TensorFlow", "PyTorch"]},
                ]
            },
        },
    }

    profile = normalize_rxresume_resume(rxresume_payload)

    assert profile.name == "Amir Tavasoli"
    assert profile.headline == "Director-level ML Engineering & Infrastructure Leader"
    assert profile.inferred_location == "Toronto, ON"
    assert profile.target_titles == [
        "Director of Data Science, Marketing",
        "Data Science Manager, Personalization",
        "Senior Data Scientist, eCommerce",
    ]
    assert profile.years_experience == 5.5
    assert "Python" in profile.top_skills
    assert "TensorFlow" in profile.top_skills
    assert "production ML platforms" in profile.summary_for_matching
    assert "Led applied AI platform strategy." in profile.summary_for_matching


def test_list_rxresume_resumes_returns_choices():
    client = FakeHttpClient(
        {
            "items": [
                {"id": "resume-1", "title": "ML Resume"},
                {"id": "resume-2", "title": "Backend Resume"},
            ]
        }
    )

    options = list_rxresume_resumes(
        "https://rx.example.com",
        "rx-key",
        http_client=client,
    )

    assert options == [
        ResumeOption(id="resume-1", label="ML Resume"),
        ResumeOption(id="resume-2", label="Backend Resume"),
    ]
    assert client.calls[0]["url"] == "https://rx.example.com/api/openapi/resumes"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer rx-key"
    assert client.calls[0]["headers"]["x-api-key"] == "rx-key"


def test_list_rxresume_resumes_supports_bare_list_payload():
    client = FakeHttpClient(
        [
            {"id": "resume-1", "name": "ML Resume"},
            {"id": "resume-2", "name": "Backend Resume"},
        ]
    )

    options = list_rxresume_resumes(
        "https://rx.example.com",
        "rx-key",
        http_client=client,
    )

    assert options == [
        ResumeOption(id="resume-1", label="ML Resume"),
        ResumeOption(id="resume-2", label="Backend Resume"),
    ]


def test_compute_years_experience_handles_iso_date_ranges_without_overcounting():
    years = _compute_years_experience(
        [
            {
                "period": "2021-01-01 - 2022-01-01",
            }
        ]
    )

    assert years == 1.0


def test_load_candidate_profile_from_pdf_uses_structured_output():
    openai_payload = {
        "name": "Grace Hopper",
        "headline": "Software Engineer",
        "inferred_location": "New York, NY",
        "target_titles": [
            "Software Engineer",
            "Platform Engineer",
            "Backend Engineer",
            "Extra Title",
        ],
        "years_experience": 9,
        "top_skills": ["Python", "Distributed Systems", "APIs"],
        "industries": ["Infrastructure"],
        "summary_for_matching": "Experienced backend engineer with API platform work.",
    }
    client = FakeOpenAIClient(openai_payload)

    profile = load_candidate_profile_from_pdf(
        b"%PDF-1.7 fake resume bytes",
        "sk-test",
        filename="resume.pdf",
        client=client,
    )

    request = client.responses.calls[0]
    input_file = request["input"][0]["content"][0]

    assert input_file["type"] == "input_file"
    assert input_file["filename"] == "resume.pdf"
    assert input_file["file_data"].startswith("data:application/pdf;base64,")
    encoded = input_file["file_data"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"%PDF-1.7 fake resume bytes"
    assert request["text"]["verbosity"] == "medium"
    assert profile.target_titles == [
        "Software Engineer",
        "Platform Engineer",
        "Backend Engineer",
    ]


def test_load_candidate_profile_from_pdf_rejects_empty_profile():
    client = FakeOpenAIClient({})

    try:
        load_candidate_profile_from_pdf(
            b"%PDF-1.7 fake resume bytes",
            "sk-test",
            filename="resume.pdf",
            client=client,
        )
    except ValueError as exc:
        assert str(exc) == "Could not extract enough resume details from the PDF."
    else:
        raise AssertionError("Expected a ValueError for an empty extracted profile")


def test_candidate_profile_response_schema_is_openai_strict():
    schema = candidate_profile_response_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "name",
        "headline",
        "inferred_location",
        "target_titles",
        "years_experience",
        "top_skills",
        "industries",
        "summary_for_matching",
    }


def test_list_rxresume_resumes_follows_redirects(monkeypatch):
    factory = RecordingHttpxFactory({"items": []})
    monkeypatch.setattr("job_finder.resume_sources.httpx.Client", factory)

    list_rxresume_resumes("https://rx.example.com", "rx-key")

    assert factory.kwargs[0]["follow_redirects"] is True


def test_build_rxresume_urls_support_default_openapi_path():
    collection_url = build_rxresume_resumes_url("https://rxresu.me/api/openapi/resumes")

    assert collection_url == "https://rxresu.me/api/openapi/resumes"
    assert build_rxresume_resume_url("https://rxresu.me/api/openapi/resumes", "resume-1") == (
        "https://rxresu.me/api/openapi/resumes/resume-1"
    )


def test_load_rxresume_resume_document_uses_openapi_resume_url():
    client = FakeHttpClient({"id": "resume-1", "data": {"basics": {"name": "Ada"}}})

    resume = load_rxresume_resume_document(
        "https://rx.example.com",
        "rx-key",
        "resume-1",
        http_client=client,
    )

    assert resume["id"] == "resume-1"
    assert client.calls[0]["url"] == "https://rx.example.com/api/openapi/resumes/resume-1"


def test_load_candidate_profile_from_rxresume_reads_openapi_data_payload():
    client = FakeHttpClient(
        {
            "id": "resume-1",
            "name": "Primary Resume",
            "data": {
                "basics": {
                    "name": "Ada Lovelace",
                    "headline": "Machine Learning Engineer",
                    "location": "Toronto, ON",
                },
                "summary": {"content": "<p>Builds production recommendation systems.</p>"},
                "sections": {
                    "experience": {
                        "items": [
                            {
                                "position": "Machine Learning Engineer",
                                "period": "Jan 2020 - Jan 2022",
                                "description": "<ul><li>Built ranking systems.</li></ul>",
                            }
                        ]
                    },
                    "skills": {
                        "items": [
                            {"name": "Core", "keywords": ["Python", "Ranking"]},
                        ]
                    },
                },
            },
        }
    )

    profile = load_candidate_profile_from_rxresume(
        "https://rx.example.com",
        "rx-key",
        "resume-1",
        http_client=client,
    )

    assert profile.name == "Ada Lovelace"
    assert profile.headline == "Machine Learning Engineer"
    assert profile.top_skills[:2] == ["Core", "Python"]


def test_import_export_delete_and_download_rxresume_artifacts(tmp_path):
    client = FakeRequestHttpClient(
        [
            {"id": "temp-resume-123"},
            {"url": "https://cdn.example.com/temp-resume-123.pdf"},
            {},
            b"%PDF-1.7 generated",
        ]
    )

    resume_id = import_rxresume_resume(
        "https://rx.example.com",
        "rx-key",
        {"basics": {"name": "Ada"}},
        name="Tailored Ada Resume",
        slug="",
        http_client=client,
    )
    pdf_url = export_rxresume_resume_pdf(
        "https://rx.example.com",
        "rx-key",
        resume_id,
        http_client=client,
    )
    delete_rxresume_resume(
        "https://rx.example.com",
        "rx-key",
        resume_id,
        http_client=client,
    )

    output_path = tmp_path / "tailored.pdf"
    download_file(pdf_url, output_path, http_client=client)

    assert resume_id == "temp-resume-123"
    assert pdf_url == "https://cdn.example.com/temp-resume-123.pdf"
    assert output_path.read_bytes() == b"%PDF-1.7 generated"
    assert client.calls[0] == {
        "method": "POST",
        "url": "https://rx.example.com/api/openapi/resumes/import",
        "headers": {
            "Authorization": "Bearer rx-key",
            "x-api-key": "rx-key",
            "Accept": "application/json",
        },
        "json": {
            "data": {"basics": {"name": "Ada"}},
            "name": "Tailored Ada Resume",
            "slug": "",
        },
        "timeout": 30.0,
    }
    assert client.calls[1]["method"] == "GET"
    assert client.calls[1]["url"] == "https://rx.example.com/api/openapi/resumes/temp-resume-123/pdf"
    assert client.calls[2]["method"] == "DELETE"
    assert client.calls[2]["url"] == "https://rx.example.com/api/openapi/resumes/temp-resume-123"
    assert client.calls[3]["method"] == "GET"
    assert client.calls[3]["url"] == "https://cdn.example.com/temp-resume-123.pdf"
