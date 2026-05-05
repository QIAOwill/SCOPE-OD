from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from model_part.engine import run_experiments
from model_part.utils import load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_GROUPS = ["Sample_Group"]


def repo_path(*parts: str, root: str | Path | None = None) -> str:
    base = Path(root).resolve() if root is not None else PROJECT_ROOT
    return str(base.joinpath(*parts))


def default_param_dict(root: str | Path | None = None, city_groups: list[str] | None = None) -> dict[str, list[Any]]:
    city_groups = city_groups or DEFAULT_CITY_GROUPS
    city_group = city_groups[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "task_type": ["single_city"],
        "group_dirs": [[repo_path("Dataset", "Data_Input", city_group, root=root)]],
        "semantic_npzs": [[repo_path("Dataset", "LLM_Outputs", city_group, "tensors", "role_semantic_tensors.npz", root=root)]],
        "target_group": [None],
        "few_shot_ratio": [0.1],
        "save_root": [repo_path("Results", "SCOPE-OD", city_group, root=root)],
        "num_history": [7],
        "future_steps": [3],
        "split_ratio": [(7, 2, 1)],
        "num_epochs": [2],
        "use_scheduler": [True],
        "hidden_dim": [16],
        "K": [2],
        "num_layers": [1],
        "kernel_type": ["chebyshev"],
        "seed": [42],
        "device": [device],
        "batch_size": [4],
        "num_workers": [0],
        "semantic_hidden_dim": [16],
        "semantic_cond_dim": [16],
        "semantic_vocab_size": [5],
        "lr": [1e-3],
        "weight_decay": [1e-5],
        "grad_clip": [1.0],
        "huber_delta": [1.0],
        "wmape_weight": [0.2],
        "best_metric": ["flow_wmape"],
        "lambda_sem_ce": [1e-4],
        "lambda_sem_origin": [1e-3],
        "lambda_sem_destination": [1e-3],
        "lambda_sem_pair": [2e-3],
        "use_semantic_alignment": [True],
        "use_regime_token": [True],
        "use_film_controller": [True],
        "use_pair_gate": [True],
        "use_uncertainty_weighting": [False],
        "uncertainty_mode": ["prob_margin"],
        "uncertainty_min_weight": [0.3],
        "use_lora_adaptation": [False],
        "lora_rank": [4],
        "lora_alpha": [8.0],
        "freeze_base_when_lora": [False],
        "max_train_windows": [8],
        "max_val_windows": [4],
        "max_test_windows": [4],
    }


def apply_city_group_paths(param_dict: dict[str, Any], city_group: str, root: str | Path | None = None) -> dict[str, Any]:
    param_dict["group_dirs"] = [[repo_path("Dataset", "Data_Input", city_group, root=root)]]
    param_dict["semantic_npzs"] = [[repo_path("Dataset", "LLM_Outputs", city_group, "tensors", "role_semantic_tensors.npz", root=root)]]
    param_dict["target_group"] = [None]
    param_dict["save_root"] = [repo_path("Results", "SCOPE-OD", city_group, root=root)]
    return param_dict


def build_param_dict(config_path: str | Path | None = None, root: str | Path | None = None, city_groups: list[str] | None = None) -> dict[str, Any]:
    base_param_dict = default_param_dict(root=root, city_groups=city_groups)
    if config_path is None:
        return base_param_dict

    override_dict = load_json(config_path)
    city_group = override_dict.pop("city_group", None)

    param_dict = deepcopy(base_param_dict)
    param_dict.update(override_dict)

    if city_group is not None:
        param_dict = apply_city_group_paths(param_dict, city_group, root=root)

    return param_dict


def configure_torch_threads() -> None:
    num_threads = int(os.environ.get("SCOPE_OD_NUM_THREADS", "1"))
    torch.set_num_threads(max(1, num_threads))


def main(config_path: str | Path | None = None, root: str | Path | None = None, city_groups: list[str] | None = None) -> None:
    configure_torch_threads()
    start_time = time.time()
    param_dict = build_param_dict(config_path=config_path, root=root, city_groups=city_groups)
    run_experiments(param_dict)
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SCOPE-OD.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON experiment config.")
    parser.add_argument("--root", type=str, default=str(PROJECT_ROOT), help="Repository root path.")
    parser.add_argument("--city-group", type=str, default=None, help="Override the default city group for no-config runs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    groups = [args.city_group] if args.city_group else None
    main(config_path=args.config, root=args.root, city_groups=groups)
