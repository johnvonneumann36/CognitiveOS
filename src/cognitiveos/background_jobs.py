from __future__ import annotations

import argparse
from pathlib import Path

from cognitiveos.config import AppSettings
from cognitiveos.service import CognitiveOSService


def _service(*, db_path: Path, memory_output_path: Path) -> CognitiveOSService:
    settings = AppSettings.from_env(
        db_path=db_path,
        memory_output_path=memory_output_path,
    )
    return CognitiveOSService.from_settings(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CognitiveOS background jobs.")
    subparsers = parser.add_subparsers(dest="job_type", required=True)

    dream_parser = subparsers.add_parser("dream-run")
    dream_parser.add_argument("--db-path", type=Path, required=True)
    dream_parser.add_argument("--memory-output-path", type=Path, required=True)
    dream_parser.add_argument("--run-id", required=True)
    dream_parser.add_argument("--window-hours", type=int, required=True)
    dream_parser.add_argument("--min-accesses", type=int, required=True)
    dream_parser.add_argument("--min-cluster-size", type=int, required=True)
    dream_parser.add_argument("--max-candidates", type=int, required=True)
    dream_parser.add_argument("--similarity-threshold", type=float, required=True)
    dream_parser.add_argument("--trigger-reason", required=True)
    dream_parser.add_argument("--auto-triggered", choices=["0", "1"], required=True)

    heuristic_parser = subparsers.add_parser("heuristic-compaction")
    heuristic_parser.add_argument("--db-path", type=Path, required=True)
    heuristic_parser.add_argument("--memory-output-path", type=Path, required=True)
    heuristic_parser.add_argument("--task-id", required=True)

    args = parser.parse_args()

    service = _service(
        db_path=args.db_path,
        memory_output_path=args.memory_output_path,
    )

    if args.job_type == "dream-run":
        service.execute_dream_run(
            run_id=args.run_id,
            output_path=args.memory_output_path,
            window_hours=args.window_hours,
            min_accesses=args.min_accesses,
            min_cluster_size=args.min_cluster_size,
            max_candidates=args.max_candidates,
            similarity_threshold=args.similarity_threshold,
            trigger_reason=args.trigger_reason,
            auto_triggered=args.auto_triggered == "1",
        )
        return

    service.execute_heuristic_compaction(task_id=args.task_id)


if __name__ == "__main__":
    main()
