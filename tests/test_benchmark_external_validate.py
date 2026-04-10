from __future__ import annotations

from benchmark.common import BENCHMARK_ROOT, load_manifest
from benchmark.scripts.validate import validate_suites


def test_benchmark_manifests_parse() -> None:
    longmemeval = load_manifest(BENCHMARK_ROOT, "longmemeval")
    locomo = load_manifest(BENCHMARK_ROOT, "locomo")
    membench = load_manifest(BENCHMARK_ROOT, "membench")

    assert longmemeval.display_name == "LongMemEval"
    assert locomo.display_name == "LoCoMo"
    assert membench.display_name == "MemBench"


def test_validate_suite_routes_and_reports_missing_data(tmp_path) -> None:
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "manifests").mkdir(parents=True)
    for name in ("longmemeval", "locomo", "membench"):
        source = BENCHMARK_ROOT / "manifests" / f"{name}.yaml"
        (benchmark_root / "manifests" / f"{name}.yaml").write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    payload = validate_suites(suite="all", benchmark_root=benchmark_root)

    assert payload["status"] == "partial"
    assert set(payload["suites"].keys()) == {"longmemeval", "locomo", "membench"}
    assert payload["suites"]["longmemeval"]["status"] == "missing"
    assert "prepare.py --suite longmemeval" in payload["suites"]["longmemeval"]["next_step"]
    assert "HighLevel and LowLevel JSON files" in payload["suites"]["membench"]["next_step"]
