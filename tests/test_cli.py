import json
from pathlib import Path

from typer.testing import CliRunner

from cognitiveos.cli.app import app

runner = CliRunner()


def test_cli_init_add_search(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.db"

    init_result = runner.invoke(app, ["init-db", "--db-path", str(db_path)])
    assert init_result.exit_code == 0

    add_result = runner.invoke(
        app,
        [
            "add",
            "--db-path",
            str(db_path),
            "--type",
            "content",
            "--payload",
            "CognitiveOS exposes a CLI and MCP server.",
            "--tag",
            "tech",
        ],
    )
    assert add_result.exit_code == 0
    node_id = json.loads(add_result.stdout)["node_id"]

    search_result = runner.invoke(
        app,
        [
            "search",
            "--db-path",
            str(db_path),
            "--keyword",
            "CognitiveOS",
        ],
    )
    assert search_result.exit_code == 0
    payload = json.loads(search_result.stdout)
    assert payload[0]["id"] == node_id

    doctor_result = runner.invoke(app, ["doctor", "--db-path", str(db_path)])
    assert doctor_result.exit_code == 0
    doctor_payload = json.loads(doctor_result.stdout)
    assert doctor_payload["sqlite_vec_version"]

    bootstrap_result = runner.invoke(
        app,
        [
            "bootstrap-host",
            "--db-path",
            str(db_path),
            "--output-dir",
            str(tmp_path / "bootstrap"),
        ],
    )
    assert bootstrap_result.exit_code == 0
    bootstrap_payload = json.loads(bootstrap_result.stdout)
    assert Path(bootstrap_payload["bootstrap_prompt_path"]).exists()

    dream_status_result = runner.invoke(
        app,
        [
            "dream",
            "--db-path",
            str(db_path),
            "--inspect",
            "status",
        ],
    )
    assert dream_status_result.exit_code == 0
    dream_status_payload = json.loads(dream_status_result.stdout)
    assert dream_status_payload["status"] == "success"
    assert "dream_status" in dream_status_payload


def test_cli_update_with_delete_tag_removes_node(tmp_path: Path) -> None:
    db_path = tmp_path / "cli-delete.db"

    init_result = runner.invoke(app, ["init-db", "--db-path", str(db_path)])
    assert init_result.exit_code == 0

    add_result = runner.invoke(
        app,
        [
            "add",
            "--db-path",
            str(db_path),
            "--type",
            "content",
            "--payload",
            "Delete me from memory.",
            "--tag",
            "temporary",
        ],
    )
    assert add_result.exit_code == 0
    node_id = json.loads(add_result.stdout)["node_id"]

    update_result = runner.invoke(
        app,
        [
            "update",
            "--db-path",
            str(db_path),
            node_id,
            "--content",
            "",
            "--tag",
            "__delete__",
        ],
    )
    assert update_result.exit_code == 0
    update_payload = json.loads(update_result.stdout)
    assert update_payload["action_taken"] == "deleted"

    read_result = runner.invoke(app, ["read", "--db-path", str(db_path), node_id])
    assert read_result.exit_code == 0
    assert node_id not in json.loads(read_result.stdout)
