from __future__ import annotations

from pathlib import Path

from cognitiveos.config import AppSettings


def _clear_runtime_env(monkeypatch) -> None:
    for name in [
        "COGNITIVEOS_HOME",
        "COGNITIVEOS_DB_PATH",
        "COGNITIVEOS_MEMORY_OUTPUT_PATH",
        "COGNITIVEOS_BOOTSTRAP_DIR",
        "COGNITIVEOS_BACKGROUND_LOG_DIR",
        "COGNITIVEOS_SNAPSHOT_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_from_env_defaults_to_shared_runtime_home_and_project_local_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    settings = AppSettings.from_env()

    shared_root = home / ".cognitiveos"
    assert settings.project_root == project.resolve()
    assert settings.db_path == shared_root / "data" / "cognitiveos.db"
    assert settings.memory_output_path == shared_root / "MEMORY.MD"
    assert settings.bootstrap_dir == project / ".cognitiveos" / "bootstrap"
    assert settings.background_log_dir == shared_root / "logs"
    assert settings.snapshot_dir == shared_root / "snapshots"


def test_from_env_project_dotenv_can_point_to_shared_runtime_home(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    shared = tmp_path / "shared-runtime"
    (project / ".env").write_text(
        "COGNITIVEOS_HOME=../shared-runtime\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    settings = AppSettings.from_env()

    assert settings.project_root == project.resolve()
    assert settings.db_path == shared / "data" / "cognitiveos.db"
    assert settings.memory_output_path == shared / "MEMORY.MD"
    assert settings.bootstrap_dir == project / ".cognitiveos" / "bootstrap"


def test_from_env_explicit_db_path_keeps_memory_in_same_runtime_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _clear_runtime_env(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    runtime = tmp_path / "runtime"
    db_path = runtime / "data" / "cognitiveos.db"
    monkeypatch.chdir(project)

    settings = AppSettings.from_env(db_path=db_path)

    assert settings.project_root == project.resolve()
    assert settings.db_path == db_path
    assert settings.memory_output_path == runtime / "MEMORY.MD"
    assert settings.bootstrap_dir == project / ".cognitiveos" / "bootstrap"
    assert settings.background_log_dir == runtime / "logs"
    assert settings.snapshot_dir == runtime / "snapshots"
