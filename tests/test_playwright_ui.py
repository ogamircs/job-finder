"""Playwright UI smoke tests for the Gradio app.

Launches the app on a random free port, verifies key UI elements render
correctly, then shuts it down.
"""

from __future__ import annotations

import multiprocessing
import socket
import time

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_app(port: int) -> None:
    from job_finder.app import build_app

    app = build_app()
    app.launch(server_port=port, share=False, prevent_thread_lock=False)


@pytest.fixture(scope="module")
def app_url():
    port = _free_port()
    proc = multiprocessing.Process(target=_run_app, args=(port,), daemon=True)
    proc.start()
    url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("App did not start in time")

    yield url
    proc.kill()
    proc.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture()
def page(browser):
    p = browser.new_page()
    p.set_viewport_size({"width": 1280, "height": 900})
    yield p
    p.close()


def _goto_app(page, app_url):
    """Navigate to the app and wait for Gradio to render."""
    page.goto(app_url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector(".gradio-container", timeout=30000)
    page.wait_for_timeout(2000)


def _click_saved_jobs_tab(page):
    """Click the Saved Jobs tab using data-tab-id selector."""
    page.locator("button[data-tab-id='saved-jobs']").click(force=True)
    page.wait_for_timeout(1000)


def _click_job_search_tab(page):
    """Click the Job Search tab using data-tab-id selector."""
    page.locator("button[data-tab-id='job-search']").click(force=True)
    page.wait_for_timeout(1000)


def _click_settings(page):
    """Click the settings toggle button using its elem_id."""
    page.locator("#search-settings-toggle").click(force=True)
    page.wait_for_timeout(500)


class TestAppLoads:
    """Verify the app loads and main elements are present."""

    def test_title_is_present(self, page, app_url):
        _goto_app(page, app_url)
        heading = page.locator("text=Resume to Jobs Finder").first
        assert heading.is_visible(), "Main heading 'Resume to Jobs Finder' not found"

    def test_job_search_tab_visible(self, page, app_url):
        _goto_app(page, app_url)
        tab = page.locator("button[data-tab-id='job-search']")
        assert tab.is_visible(), "Job Search tab not visible"

    def test_saved_jobs_tab_visible(self, page, app_url):
        _goto_app(page, app_url)
        tab = page.locator("button[data-tab-id='saved-jobs']")
        assert tab.is_visible(), "Saved Jobs tab not visible"

    def test_settings_button_visible(self, page, app_url):
        _goto_app(page, app_url)
        btn = page.locator("#search-settings-toggle")
        assert btn.is_visible(), "Settings button not visible"

    def test_resume_source_radio(self, page, app_url):
        _goto_app(page, app_url)
        pdf_radio = page.locator("text=PDF").first
        assert pdf_radio.is_visible(), "PDF radio option not found"

    def test_reactive_resume_radio(self, page, app_url):
        _goto_app(page, app_url)
        rx_radio = page.locator("text=Reactive Resume").first
        assert rx_radio.is_visible(), "Reactive Resume radio option not found"

    def test_analyze_resume_button(self, page, app_url):
        _goto_app(page, app_url)
        btn = page.locator("button:has-text('Analyze resume')")
        assert btn.is_visible(), "Analyze resume button not found"

    def test_find_jobs_button(self, page, app_url):
        _goto_app(page, app_url)
        btn = page.locator("button:has-text('Find jobs')")
        assert btn.is_visible(), "Find jobs button not found"

    def test_location_input(self, page, app_url):
        _goto_app(page, app_url)
        loc_input = page.locator("label:has-text('Location')")
        assert loc_input.is_visible(), "Location input not found"

    def test_include_remote_checkbox(self, page, app_url):
        _goto_app(page, app_url)
        checkbox = page.locator("text=Include remote jobs")
        assert checkbox.is_visible(), "Include remote jobs checkbox not found"

    def test_search_terms_textarea(self, page, app_url):
        _goto_app(page, app_url)
        textarea = page.locator("label:has-text('Search terms')")
        assert textarea.is_visible(), "Search terms textarea not found"

    def test_results_table_present(self, page, app_url):
        _goto_app(page, app_url)
        table = page.locator("#job-results-table")
        assert table.is_visible(), "Results table not found"

    def test_sort_dropdown_present(self, page, app_url):
        _goto_app(page, app_url)
        # Gradio renders dropdown labels as span inside label elements
        sort = page.locator("text=Sort by").first
        assert sort.is_visible(), "Sort by dropdown not found"

    def test_filter_input(self, page, app_url):
        _goto_app(page, app_url)
        filter_input = page.locator("text=Filter title or company").first
        assert filter_input.is_visible(), "Filter input not found"


class TestStepCards:
    """Verify the step card structure."""

    def test_step_1_resume_source(self, page, app_url):
        _goto_app(page, app_url)
        step = page.locator("text=1. Resume Source").first
        assert step.is_visible(), "Step 1 Resume Source not visible"

    def test_step_2_analyze_resume(self, page, app_url):
        _goto_app(page, app_url)
        step = page.locator("text=2. Analyze Resume").first
        assert step.is_visible(), "Step 2 Analyze Resume not visible"

    def test_step_3_search_preferences(self, page, app_url):
        _goto_app(page, app_url)
        step = page.locator("text=3. Search Preferences").first
        assert step.is_visible(), "Step 3 Search Preferences not visible"

    def test_step_4_results(self, page, app_url):
        _goto_app(page, app_url)
        step = page.locator("text=4. Results").first
        assert step.is_visible(), "Step 4 Results not visible"

    def test_step_1_complete_state(self, page, app_url):
        _goto_app(page, app_url)
        badge = page.locator(".step-pill.complete").first
        assert badge.is_visible(), "Step 1 Complete badge not visible"

    def test_step_2_current_state(self, page, app_url):
        _goto_app(page, app_url)
        badge = page.locator(".step-pill.current").first
        assert badge.is_visible(), "Step 2 Current badge not visible"

    def test_status_banner_visible(self, page, app_url):
        _goto_app(page, app_url)
        banner = page.locator(".status-banner").first
        assert banner.is_visible(), "Status banner not visible"


class TestResumeSourceSection:
    """Verify the resume source section."""

    def test_saved_pdf_dropdown(self, page, app_url):
        _goto_app(page, app_url)
        dropdown = page.locator("text=Saved PDF resumes").first
        assert dropdown.is_visible(), "Saved PDF resumes dropdown not visible"

    def test_upload_area(self, page, app_url):
        _goto_app(page, app_url)
        upload = page.locator("text=Drop File Here").first
        assert upload.is_visible(), "Upload area not visible"

    def test_selected_resume_shown(self, page, app_url):
        _goto_app(page, app_url)
        source_label = page.locator("text=Selected resume source").first
        assert source_label.is_visible(), "Selected resume source label not visible"

    def test_resume_file_displayed(self, page, app_url):
        _goto_app(page, app_url)
        resume = page.locator("text=Amir_Tavasoli_Resume_2026.pdf").first
        assert resume.is_visible(), "Resume filename not displayed"


class TestAnalyzeFlow:
    """Verify analysis updates the workspace instead of resetting it."""

    def test_analyze_populates_profile_and_enables_search(self, page, app_url):
        _goto_app(page, app_url)
        page.locator("button:has-text('Analyze resume')").click()
        page.locator("text=Resume analyzed.").first.wait_for(timeout=60000)

        summary = page.locator(".summary-card").nth(1)
        assert "No analysis yet" not in summary.text_content()

        location = page.get_by_role("textbox", name="Location")
        assert location.input_value().strip(), "Location should be populated after analysis"

        search_terms = page.get_by_role("textbox", name="Search terms")
        assert search_terms.input_value().strip(), "Search terms should be populated after analysis"

        find_jobs = page.get_by_role("button", name="Find jobs")
        assert find_jobs.is_enabled(), "Find jobs should be enabled after analysis"


class TestSettingsPanel:
    """Verify the settings panel toggles correctly."""

    def test_settings_panel_initially_hidden(self, page, app_url):
        _goto_app(page, app_url)
        # Settings panel should not be visible initially
        panel = page.locator("#settings-panel")
        assert not panel.is_visible(), "Settings panel should be hidden initially"

    def test_settings_toggle_opens(self, page, app_url):
        _goto_app(page, app_url)
        _click_settings(page)
        # After clicking, OpenAI API key field should appear
        openai_key = page.locator("text=OpenAI API key")
        assert openai_key.count() >= 1, "OpenAI API key field not visible after toggle"

    def test_settings_has_save_button(self, page, app_url):
        _goto_app(page, app_url)
        _click_settings(page)
        btn = page.locator("button:has-text('Save settings')")
        assert btn.is_visible(), "Save settings button not visible"

    def test_settings_fields_present(self, page, app_url):
        _goto_app(page, app_url)
        _click_settings(page)
        assert page.locator("text=SerpApi key").first.is_visible(), "SerpApi key not visible"
        assert page.locator("text=Backend model").first.is_visible(), "Backend model not visible"


class TestNavigation:
    """Verify navigation between Job Search and Saved Jobs."""

    def test_switch_to_saved_jobs(self, page, app_url):
        _goto_app(page, app_url)
        _click_saved_jobs_tab(page)
        table = page.locator("#saved-jobs-table")
        assert table.is_visible(), "Saved jobs table not visible after switching"

    def test_switch_back_to_job_search(self, page, app_url):
        _goto_app(page, app_url)
        _click_saved_jobs_tab(page)
        _click_job_search_tab(page)
        table = page.locator("#job-results-table")
        assert table.is_visible(), "Results table not visible after navigating back"

    def test_saved_jobs_shows_edit_fields(self, page, app_url):
        _goto_app(page, app_url)
        _click_saved_jobs_tab(page)
        title_field = page.locator("label:has-text('Title')")
        assert title_field.count() >= 1, "Title field not visible in saved jobs view"


class TestResultsTableHeaders:
    """Verify the results table has correct column headers."""

    def test_score_column(self, page, app_url):
        _goto_app(page, app_url)
        assert page.locator("#job-results-table >> text=Score").first.is_visible()

    def test_title_column(self, page, app_url):
        _goto_app(page, app_url)
        assert page.locator("#job-results-table >> text=Title").first.is_visible()

    def test_company_column(self, page, app_url):
        _goto_app(page, app_url)
        assert page.locator("#job-results-table >> text=Company").first.is_visible()

    def test_location_column(self, page, app_url):
        _goto_app(page, app_url)
        assert page.locator("#job-results-table >> text=Location").first.is_visible()

    def test_apply_column(self, page, app_url):
        _goto_app(page, app_url)
        assert page.locator("#job-results-table >> text=Apply").first.is_visible()


class TestScreenshots:
    """Take screenshots for visual verification."""

    def test_main_view_screenshot(self, page, app_url):
        _goto_app(page, app_url)
        page.screenshot(
            path="/Users/ymir/git_repos/job-finder/tests/screenshot_main.png",
            full_page=True,
        )

    def test_saved_jobs_screenshot(self, page, app_url):
        _goto_app(page, app_url)
        _click_saved_jobs_tab(page)
        page.screenshot(
            path="/Users/ymir/git_repos/job-finder/tests/screenshot_saved_jobs.png",
            full_page=True,
        )

    def test_settings_screenshot(self, page, app_url):
        _goto_app(page, app_url)
        _click_settings(page)
        page.screenshot(
            path="/Users/ymir/git_repos/job-finder/tests/screenshot_settings.png",
            full_page=True,
        )
