from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schemas import stage_keys


@dataclass
class GroupDataset:

    group_name: str
    city_static: pd.DataFrame
    pair_static: pd.DataFrame
    od: pd.DataFrame


def load_group_dataset(group_dir: str | Path) -> GroupDataset:
    group_dir = Path(group_dir)
    group = group_dir.name

    city_static = pd.read_csv(group_dir / f"{group}_city_static.txt", sep="	")
    pair_static = pd.read_csv(group_dir / f"{group}_city_pair_static.txt", sep="	")
    od = pd.read_csv(group_dir / f"{group}_OD.txt", sep="	")


    city_static["urban_agglomeration"] = group
    pair_static["urban_agglomeration"] = group


    od["date"] = pd.to_datetime(od["date"])

    return GroupDataset(group, city_static, pair_static, od)


def _ensure_probability_columns(df: pd.DataFrame, semantic_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in semantic_cols:
        for level in range(5):
            prob_col = f"{col}_prob_{level}"
            if prob_col not in out.columns:
                out[prob_col] = 0.0
            out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce").fillna(0.0)
    return out


def _ensure_conf_entropy_columns(df: pd.DataFrame, semantic_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in semantic_cols:
        conf_col = f"{col}_confidence"
        ent_col = f"{col}_entropy"
        if conf_col not in out.columns:
            out[conf_col] = 0.5
        if ent_col not in out.columns:
            out[ent_col] = 1.0
        out[conf_col] = pd.to_numeric(out[conf_col], errors="coerce").fillna(0.5)
        out[ent_col] = pd.to_numeric(out[ent_col], errors="coerce").fillna(1.0)
    return out


def build_dense_outputs(
    group_data: GroupDataset,
    origin_city_static_sem: pd.DataFrame,
    destination_city_static_sem: pd.DataFrame,
    pair_relation_static_sem: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    city_ids = sorted(group_data.city_static["city_id"].astype(int).tolist())
    city_id_to_idx = {cid: idx for idx, cid in enumerate(city_ids)}

    ocs_cols = stage_keys("origin_city_static")
    dcs_cols = stage_keys("destination_city_static")
    prs_cols = stage_keys("pair_relation_static")

    n = len(city_ids)
    prob_bins = 5


    origin_city_static_arr = np.zeros((n, len(ocs_cols)), dtype=np.int16)
    destination_city_static_arr = np.zeros((n, len(dcs_cols)), dtype=np.int16)
    pair_relation_static_arr = np.zeros((n, n, len(prs_cols)), dtype=np.int16)
    pair_mask = np.zeros((n, n), dtype=np.int8)


    origin_city_static_prob = np.zeros((n, len(ocs_cols), prob_bins), dtype=np.float32)
    destination_city_static_prob = np.zeros((n, len(dcs_cols), prob_bins), dtype=np.float32)
    pair_relation_static_prob = np.zeros((n, n, len(prs_cols), prob_bins), dtype=np.float32)

    origin_city_static_conf = np.zeros((n, len(ocs_cols)), dtype=np.float32)
    destination_city_static_conf = np.zeros((n, len(dcs_cols)), dtype=np.float32)
    pair_relation_static_conf = np.zeros((n, n, len(prs_cols)), dtype=np.float32)

    origin_city_static_entropy = np.ones((n, len(ocs_cols)), dtype=np.float32)
    destination_city_static_entropy = np.ones((n, len(dcs_cols)), dtype=np.float32)
    pair_relation_static_entropy = np.ones((n, n, len(prs_cols)), dtype=np.float32)


    ocs = _ensure_conf_entropy_columns(
    _ensure_probability_columns(origin_city_static_sem.copy(), ocs_cols), ocs_cols
    )
    ocs["city_idx"] = ocs["city_id"].astype(int).map(city_id_to_idx)
    valid = ocs["city_idx"].notna()
    ocs = ocs.loc[valid].copy()
    city_idx = ocs["city_idx"].to_numpy(dtype=np.int64)


    origin_city_static_arr[city_idx] = ocs[ocs_cols].fillna(0).to_numpy(dtype=np.int16)
    for fi, col in enumerate(ocs_cols):
        origin_city_static_conf[city_idx, fi] = ocs[f"{col}_confidence"].to_numpy(dtype=np.float32)
        origin_city_static_entropy[city_idx, fi] = ocs[f"{col}_entropy"].to_numpy(dtype=np.float32)
        prob_cols = [f"{col}_prob_{level}" for level in range(prob_bins)]
        origin_city_static_prob[city_idx, fi, :] = ocs[prob_cols].to_numpy(dtype=np.float32)


    dcs = _ensure_conf_entropy_columns(
    _ensure_probability_columns(destination_city_static_sem.copy(), dcs_cols), dcs_cols
    )
    dcs["city_idx"] = dcs["city_id"].astype(int).map(city_id_to_idx)
    valid = dcs["city_idx"].notna()
    dcs = dcs.loc[valid].copy()
    city_idx = dcs["city_idx"].to_numpy(dtype=np.int64)

    destination_city_static_arr[city_idx] = dcs[dcs_cols].fillna(0).to_numpy(dtype=np.int16)
    for fi, col in enumerate(dcs_cols):
        destination_city_static_conf[city_idx, fi] = dcs[f"{col}_confidence"].to_numpy(dtype=np.float32)
        destination_city_static_entropy[city_idx, fi] = dcs[f"{col}_entropy"].to_numpy(dtype=np.float32)
        prob_cols = [f"{col}_prob_{level}" for level in range(prob_bins)]
        destination_city_static_prob[city_idx, fi, :] = dcs[prob_cols].to_numpy(dtype=np.float32)


    prs = _ensure_conf_entropy_columns(
    _ensure_probability_columns(pair_relation_static_sem.copy(), prs_cols), prs_cols
    )
    prs["origin_idx"] = prs["origin_id"].astype(int).map(city_id_to_idx)
    prs["destination_idx"] = prs["destination_id"].astype(int).map(city_id_to_idx)
    valid = prs["origin_idx"].notna() & prs["destination_idx"].notna()
    prs = prs.loc[valid].copy()
    origin_idx = prs["origin_idx"].to_numpy(dtype=np.int64)
    dest_idx = prs["destination_idx"].to_numpy(dtype=np.int64)

    pair_relation_static_arr[origin_idx, dest_idx] = prs[prs_cols].fillna(0).to_numpy(dtype=np.int16)

    pair_mask[origin_idx, dest_idx] = 1
    for fi, col in enumerate(prs_cols):
        pair_relation_static_conf[origin_idx, dest_idx, fi] = prs[f"{col}_confidence"].to_numpy(dtype=np.float32)
        pair_relation_static_entropy[origin_idx, dest_idx, fi] = prs[f"{col}_entropy"].to_numpy(dtype=np.float32)
        prob_cols = [f"{col}_prob_{level}" for level in range(prob_bins)]
        pair_relation_static_prob[origin_idx, dest_idx, fi, :] = prs[prob_cols].to_numpy(dtype=np.float32)


    city_static_arr = np.concatenate([origin_city_static_arr, destination_city_static_arr], axis=-1)

    arrays = {


        "origin_city_static_sem": origin_city_static_arr,
        "destination_city_static_sem": destination_city_static_arr,
        "pair_relation_static_sem": pair_relation_static_arr,
        "pair_mask": pair_mask,
        "city_static_sem": city_static_arr,
        "pair_static_sem": pair_relation_static_arr,


        "origin_city_static_sem_prob": origin_city_static_prob,
        "destination_city_static_sem_prob": destination_city_static_prob,
        "pair_relation_static_sem_prob": pair_relation_static_prob,

        "origin_city_static_sem_confidence": origin_city_static_conf,
        "destination_city_static_sem_confidence": destination_city_static_conf,
        "pair_relation_static_sem_confidence": pair_relation_static_conf,

        "origin_city_static_sem_entropy": origin_city_static_entropy,
        "destination_city_static_sem_entropy": destination_city_static_entropy,
        "pair_relation_static_sem_entropy": pair_relation_static_entropy,
    }

    meta = {
        "city_id_to_index": {str(k): int(v) for k, v in city_id_to_idx.items()},
        "feature_names": {
            "origin_city_static_sem": ocs_cols,
            "destination_city_static_sem": dcs_cols,
            "pair_relation_static_sem": prs_cols,
            "city_static_sem": [f"origin::{c}" for c in ocs_cols] + [f"destination::{c}" for c in dcs_cols],
            "pair_static_sem": prs_cols,
        },
        "probability_bins": [0, 1, 2, 3, 4],
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    return arrays, meta
