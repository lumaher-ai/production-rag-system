"""Config file paths must not depend on the process's working directory.

A relative path in ``.env`` that resolves under ``uvicorn`` started in the repo
root but not under Docker or a test runner is the classic "works on my machine"
failure — and it surfaces as "not configured", which points at the wrong cause.
"""

from pathlib import Path

import pytest

from production_rag.config import PROJECT_ROOT, Settings, resolve_config_path


def test_project_root_is_the_repository_root() -> None:
    """Anchored on a file that must exist at the root, not on a path guess."""
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "production_rag").is_dir()


def test_empty_stays_empty() -> None:
    """Empty means 'not configured' — a valid state, not a path to resolve."""
    assert resolve_config_path("") == ""


def test_absolute_path_is_preserved(tmp_path: Path) -> None:
    key = tmp_path / "key.json"
    key.write_text("{}")

    assert resolve_config_path(str(key)) == str(key)


def test_absolute_path_preserved_even_when_missing(tmp_path: Path) -> None:
    """An explicit absolute path is never second-guessed; the error names it."""
    missing = str(tmp_path / "absent.json")

    assert resolve_config_path(missing) == missing


def test_relative_path_resolves_against_cwd_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "secrets").mkdir()
    key = tmp_path / "secrets" / "key.json"
    key.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert resolve_config_path("secrets/key.json") == str(key.resolve())


def test_relative_path_falls_back_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix: launching from elsewhere still finds a repo-relative key."""
    secrets = PROJECT_ROOT / "secrets"
    created = not secrets.exists()
    secrets.mkdir(exist_ok=True)
    key = secrets / "_resolve_probe.json"
    key.write_text("{}")
    # Somewhere with no 'secrets/' of its own, as Docker or systemd would be.
    monkeypatch.chdir(tmp_path)

    try:
        assert resolve_config_path("secrets/_resolve_probe.json") == str(key.resolve())
    finally:
        key.unlink()
        if created:
            secrets.rmdir()


def test_missing_relative_path_reports_a_concrete_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unresolvable paths become absolute so the error names a real location."""
    monkeypatch.chdir(tmp_path)

    resolved = resolve_config_path("secrets/nope.json")

    assert Path(resolved).is_absolute()
    assert resolved.endswith("secrets/nope.json")


def test_settings_resolves_the_service_account_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validator runs on load, so every consumer sees the same path."""
    (tmp_path / "secrets").mkdir()
    key = tmp_path / "secrets" / "sa.json"
    key.write_text("{}")
    monkeypatch.chdir(tmp_path)

    settings = Settings(google_service_account_file="secrets/sa.json")

    assert settings.google_service_account_file == str(key.resolve())


def test_settings_leaves_unconfigured_service_account_empty() -> None:
    assert Settings(google_service_account_file="").google_service_account_file == ""


def test_no_setting_is_required() -> None:
    """CI checks out the repo and runs the suite with no ``.env`` at all.

    That works today only because every field carries a default. Adding one
    without a default makes CI red, a fresh clone unrunnable, and any deploy
    that forgot the variable fail — and it fails *far from its cause*, wherever
    ``get_settings()`` is first called rather than at the line that introduced
    it. This pins the property so the break names the field instead.
    """
    required = sorted(name for name, f in Settings.model_fields.items() if f.is_required())

    assert required == [], (
        "These settings have no default, so anything without a .env cannot start: "
        f"{', '.join(required)}. Give each one a default (empty string means "
        "'not configured', which is a valid state here), or teach CI to supply it."
    )


def test_settings_load_from_a_directory_with_no_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end form of the above: ``env_file`` is resolved per CWD.

    Constructing from an empty directory is what a CI runner does. Kept separate
    from the field check because this one can be masked by variables exported in
    a developer's shell, while the field check cannot.
    """
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.app_name
    assert settings.database_url
