from __future__ import annotations

from itertools import product
from typing import Any, Dict, List


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return [value]


def list_of_param_dicts(param_dict: Dict[str, Any]) -> List[Dict[str, Any]]:

    normalized = {k: _ensure_list(v) for k, v in param_dict.items()}

    keys = list(normalized.keys())
    values = [normalized[k] for k in keys]


    return [dict(zip(keys, combo)) for combo in product(*values)]


class Args:

    def __init__(self, arg_dict: Dict[str, Any]):
        self.__dict__.update(arg_dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
