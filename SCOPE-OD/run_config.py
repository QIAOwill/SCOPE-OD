from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from main import PROJECT_ROOT, main

DEFAULT_CONFIGS = ["Sample_Group/01_quickstart.json"]


def run_config_list(config_list: Sequence[str], config_dir: str | Path = "configs", root: str | Path | None = None) -> None:
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = Path(__file__).resolve().parent / config_dir

    for idx, config_name in enumerate(config_list, start=1):
        config_path = config_dir / config_name
        if config_path.suffix == "":
            config_path = config_path.with_suffix(".json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"\n[{idx}/{len(config_list)}] Running {config_path}")
        main(config_path=config_path, root=root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more SCOPE-OD config files.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start = time.time()
    run_config_list(args.configs, config_dir=args.config_dir, root=args.root)
    print(f"Finished config list in {time.time() - start:.2f} seconds.")
