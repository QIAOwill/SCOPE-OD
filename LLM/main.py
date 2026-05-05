from __future__ import annotations

import argparse
import time
from pathlib import Path

from LLM_Semantic.io_utils import list_group_dirs
from LLM_Semantic.pipeline import DualSemanticPipeline, PipelineArgs
from LLM_Semantic.runner import run_with_auto_restart

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(data_dir: str | Path, output_dir: str | Path, config: str | Path, groups: list[str] | None = None, force: bool = False) -> None:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    group_names = groups or [p.name for p in list_group_dirs(data_dir)]
    pipe_args = PipelineArgs(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        config_path=str(config),
        groups=group_names,
        skip_existing=not force,
    )
    DualSemanticPipeline(pipe_args).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCOPE-OD role-aware semantic priors.")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "Dataset" / "Data_Input"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "Dataset" / "LLM_Outputs"))
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "API_Config.json"))
    parser.add_argument("--groups", nargs="*", default=["Sample_Group"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--auto-restart", action="store_true")
    parser.add_argument("--restart-wait-seconds", type=int, default=5)
    parser.add_argument("--max-restarts", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    def job() -> None:
        main(args.data_dir, args.output_dir, args.config, args.groups, args.force)

    if args.auto_restart:
        run_with_auto_restart(
            target=job,
            restart_wait_seconds=args.restart_wait_seconds,
            max_run_restarts=args.max_restarts,
            log_path=Path(args.output_dir) / "_run_logs" / "auto_restart.log",
        )
    else:
        job()

    print(f"Finished semantic prior generation in {time.time() - start:.2f} seconds.")
