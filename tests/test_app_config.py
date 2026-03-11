from job_finder.app import build_app


def _component_by_label(app, label):
    for component in app.config["components"]:
        if component["props"].get("label") == label:
            return component
    raise AssertionError(f"Component with label {label!r} not found")


def _component_by_value(app, value):
    for component in app.config["components"]:
        if component["props"].get("value") == value:
            return component
    raise AssertionError(f"Component with value {value!r} not found")


def _first_component_by_type(app, component_type):
    for component in app.config["components"]:
        if component.get("type") == component_type:
            return component
    raise AssertionError(f"Component with type {component_type!r} not found")


def test_build_app_sets_rxresume_default_endpoint():
    app = build_app()
    component = _component_by_label(app, "Reactive Resume API URL")

    assert component["props"]["value"] == "https://rxresu.me/api/openapi/resumes"


def test_build_app_starts_with_find_jobs_disabled_and_supports_browse_and_settings():
    app = build_app()

    find_jobs_button = _component_by_value(app, "Find jobs")
    _component_by_value(app, "Settings")
    _component_by_label(app, "Browse PDF")

    assert find_jobs_button["props"]["interactive"] is False


def test_build_app_adds_pay_range_to_results_table():
    app = build_app()
    dataframe = _first_component_by_type(app, "dataframe")

    assert "pay_range" in dataframe["props"]["headers"]


def test_build_app_disables_queue_for_resume_actions():
    app = build_app()
    dependencies = {dependency["api_name"]: dependency for dependency in app.config["dependencies"]}

    assert dependencies["save_setup_ui"]["queue"] is False
    assert dependencies["preview_resume_ui"]["queue"] is False
    assert dependencies["find_jobs_ui"]["queue"] is False
