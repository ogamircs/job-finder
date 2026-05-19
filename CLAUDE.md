# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# install deps (Python 3.11)
uv sync --python 3.11 --all-groups

# launch Gradio app (entrypoint: src/job_finder/__main__.py)
uv run job-finder

# run full test suite
uv run pytest

# run a single test file / test
uv run pytest tests/test_matching.py
uv run pytest tests/test_matching.py::test_dedupe_jobs

# lint / format (ruff installed in .venv, not declared in pyproject)
uv run ruff check .
uv run ruff format .
```

Playwright UI tests (`tests/test_playwright_ui.py`) spawn the app via `multiprocessing` on a free port and drive Chromium headless. They require `playwright` + browser binaries (`uv run playwright install chromium`). Not declared in `pyproject.toml` dependency groups — install ad hoc if you need them.

`pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`, so tests import `job_finder.*` without an editable install.

## Architecture

Single-process Gradio app. Resume → candidate profile → SerpApi Google Jobs search → OpenAI scoring → ranked matches → optional SQLite save → optional tailored-resume artifact generation via Reactive Resume.

### Module layout (`src/job_finder/`)

- `app.py` (~5.2k LOC) — Gradio UI + orchestration. Contains three layers:
  - `JobMatchService` — pure business logic, injectable loaders/provider/matcher. No Gradio.
  - `AppController` — thin coordinator that talks to `LocalWorkspace`, `SavedJobsStore`, and `JobMatchService`. Returns plain dicts.
  - `build_app(controller, workspace)` — composes the Gradio Blocks UI (wizard flow, settings panel, results table, saved-jobs tab). `_legacy_build_app` is the older flat layout kept for reference.
- `models.py` — Pydantic v2 models. `CandidateProfile`, `JobPosting`, `ScoredJobMatch`, `SearchRequest`, `SearchRunResult`, `SavedJobRecord`, `TailoredApplicationContent`, `GeneratedApplicationArtifacts`. Validators normalize strings/lists, dedupe case-insensitively, cap `target_titles` at 3.
- `workspace.py` — `LocalWorkspace` owns `.env` read/write, `.resume/` PDF storage, and the SQLite path. `resolve_value(key, explicit)` precedence: explicit arg → `os.environ` → `.env`. Only keys in `ENV_KEYS` are persisted.
- `resume_sources.py` — Two profile loaders: `load_candidate_profile_from_pdf` (sends base64 PDF to OpenAI Responses API with structured JSON schema) and `load_candidate_profile_from_rxresume` (fetches Reactive Resume JSON, `normalize_rxresume_resume` flattens sections/computes `years_experience` from experience date ranges). Also wraps RxResume REST calls used by application_documents.
- `job_provider.py` — Abstract `JobProvider` + `SerpApiGoogleJobsProvider`. `_normalize_serpapi_location` expands `City, ST` → full country form. `parse_serpapi_job` filters: drops jobs missing title/company/apply_url. Remote detection inspects location and `detected_extensions.schedule_type`.
- `matching.py` — Pipeline used by `JobMatchService.run_search`:
  1. `build_search_queries` (user terms → profile titles → headline → "Software Engineer"), capped at 3.
  2. For each query, hit provider with location + remote variants → `dedupe_jobs` (provider+id or title+company+url signature).
  3. `prefilter_jobs` heuristic rank, take top 25.
  4. `score_jobs` calls OpenAI Responses API in batches of 5 with strict JSON schema (`score_10`, `rationale`, `matched_skills`, `missing_signals`).
  5. `finalize_matches` filters by `threshold` (default 7) and trims to `result_limit` (default 15).
- `saved_jobs.py` — SQLite store at `.saved_jobs.sqlite3`. `saved_job_identity` builds the dedupe key: prefer `provider:provider_job_id`, else fall back to `provider|title|company|apply_url` signature. **Include `provider` in fallback to avoid cross-provider collisions** (see commit `4af87d5`). Save is idempotent — `UNIQUE(dedupe_key)` triggers the "already saved" branch.
- `application_documents.py` — `ApplicationArtifactsService.generate_application_artifacts` flow: load remote RxResume → call OpenAI for `TailoredApplicationContent` → `apply_tailored_resume_content` merges into a clone (preserves `sections.skills` item templates) → re-import to RxResume → export PDF → download to `output/generated/<timestamp>-<company>-<title>/` → **best-effort delete** of the remote import in `finally`. All collaborators are injectable for tests.

### Data flow

`Gradio event → AppController method → JobMatchService.run_search (or preview/generate) → returns SearchRunResult/dict → Gradio state updates → table/cards re-render`.

State lives in Gradio `gr.State` blobs (matches, profile, generated_artifacts, etc.) — `build_app` uses dicts as state payloads, not class instances, because Gradio serializes them. Pydantic models go through `model_dump()` before entering state and `model_validate()` on exit.

### External APIs

- **OpenAI Responses API** (`openai>=1.68`) using `text.format = json_schema` strict mode. Default model `gpt-4o`, overridable via `OPENAI_MODEL`.
- **SerpApi** `engine=google_jobs`.
- **Reactive Resume** OpenAPI: `GET /api/openapi/resumes`, `GET /:id`, `POST /import`, `GET /:id/pdf`, `DELETE /:id`. Auth sent as both `Authorization: Bearer` and `x-api-key` to handle either deployment.

### Test conventions

Pure-logic modules (`matching`, `job_provider`, `resume_sources`, `application_documents`, `saved_jobs`, `workspace`) test directly. `test_service.py` exercises `JobMatchService` with fake loaders/provider/matcher injected — match this pattern when adding service logic. `test_app_config.py` smoke-imports `build_app`. `test_playwright_ui.py` is the only test that boots the actual server.

## Conventions

- Pydantic v2 validators are `mode="before"` and normalize whitespace/case. When adding fields, follow the existing `_clean_text` / `_clean_unique_strings` style.
- Functions that touch external services accept an optional `http_client` / `client` kwarg for test injection — keep this pattern.
- The `pay_range` field is parsed heuristically (`_PAY_PATTERN` + digit check) from multiple SerpApi locations; do not assume it's populated.
