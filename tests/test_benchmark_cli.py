import json
from pathlib import Path

from typer.testing import CliRunner

from cognitiveos.cli.app import app

runner = CliRunner()


def test_benchmark_cli_writes_report(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    output_path = tmp_path / "benchmark-report.json"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--iterations",
            "1",
            "--provider-mode",
            "fake",
            "--runtime-dir",
            str(runtime_dir),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["suite_name"] == "github_readiness_baseline"
    assert payload["quality"]["total_tasks"] >= 6
    assert payload["quality"]["passed_tasks"] >= 4
    assert payload["runtime"]["operations"]["search_keyword"]["iterations"] == 1
    assert payload["dream"]["status"] in {"success", "awaiting_host_compaction"}
    assert output_path.exists()
