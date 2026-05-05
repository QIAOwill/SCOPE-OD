from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_json_atomic(obj: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target)


def append_jsonl_row(path: str | Path, row: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_text_line(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_jsonl_as_dict(path: str | Path, key_field: str = "cache_key") -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return result

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if key_field not in row:
                continue
            result[row[key_field]] = row
    return result


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def extract_first_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No parseable JSON found: {text[:200]}")
    return json.loads(match.group(0))


def list_group_dirs(data_dir: str | Path) -> List[Path]:
    base = Path(data_dir)
    group_dirs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(group_dirs, key=lambda x: x.name)

def normalized_entropy(probabilities: List[float]) -> float:

    arr = np.asarray(probabilities, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:

        return 1.0


    arr = arr / total
    eps = 1e-12


    ent = -float(np.sum(arr * np.log(arr + eps)))
    max_ent = math.log(len(arr)) if len(arr) > 1 else 1.0
    if max_ent <= 0:
        return 0.0


    entropy = float(ent / max_ent)


    if abs(entropy) < 1e-12:
        entropy = 0.0


    entropy = max(0.0, min(1.0, entropy))

    return entropy
