from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def merge_dict_of_scalars(dst: Dict[str, float], src: Dict[str, float], weight: float = 1.0) -> Dict[str, float]:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value) * weight
    return dst


def normalize_metrics(values: Dict[str, float], denom: float) -> Dict[str, float]:
    denom = max(float(denom), 1.0)
    return {key: float(value) / denom for key, value in values.items()}


def city_group_name(index: int, city_groups: list[str] | None = None) -> str:
    city_groups = city_groups or ["Sample_Group"]
    if 0 <= index < len(city_groups):
        return city_groups[index]
    raise ValueError(f"Invalid city group index: {index}. Valid range: 0 to {len(city_groups) - 1}.")


def get_root_path(run_device: str | None = None, device_path: dict[str, str] | None = None) -> str:
    if device_path and run_device in device_path:
        return device_path[run_device]
    return str(Path(__file__).resolve().parents[2])
