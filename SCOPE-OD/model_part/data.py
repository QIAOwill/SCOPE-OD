from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class GroupDataBundle:

    group_name: str
    domain_id: int
    city_ids: List[int]
    dates: List[str]
    od_raw: np.ndarray
    od_log: np.ndarray
    od_obs_mask: np.ndarray
    base_adj: np.ndarray
    sym_adj: np.ndarray
    graph_kernel: np.ndarray
    regime: np.ndarray
    pair_static_num: np.ndarray
    node_mask: np.ndarray
    pair_mask: np.ndarray
    sem: Dict[str, np.ndarray]


class AdjProcessor:

    def __init__(self, kernel_type: str, K: int):
        self.kernel_type = kernel_type
        self.K = int(K)

    def process(self, adj: np.ndarray) -> np.ndarray:
        a = torch.from_numpy(adj).float()


        if self.kernel_type == "chebyshev":
            a_norm = self.symmetric_normalize(a)
            lap = torch.eye(a.shape[0]) - a_norm
            lap = self.rescale_laplacian(lap)

            kernels: List[torch.Tensor] = []
            for k in range(self.K):
                if k == 0:

                    kernels.append(torch.eye(a.shape[0]))
                elif k == 1:

                    kernels.append(lap)
                else:

                    kernels.append(2 * lap @ kernels[k - 1] - kernels[k - 2])

            return torch.stack(kernels, dim=0).numpy().astype(np.float32)


        if self.kernel_type == "random_walk_diffusion":

            rw = self.random_walk_normalize(a)
            kernels = []
            for k in range(self.K):
                if k == 0:
                    kernels.append(torch.eye(a.shape[0]))
                elif k == 1:
                    kernels.append(rw.T)
                else:
                    kernels.append(rw.T @ kernels[k - 1])

            return torch.stack(kernels, dim=0).numpy().astype(np.float32)

        raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

    @staticmethod
    def random_walk_normalize(a: torch.Tensor) -> torch.Tensor:
        deg = a.sum(dim=1)
        deg = torch.where(deg > 0, deg, torch.ones_like(deg))
        d_inv = torch.diag(torch.pow(deg, -1.0))
        return d_inv @ a

    @staticmethod
    def symmetric_normalize(a: torch.Tensor) -> torch.Tensor:
        deg = a.sum(dim=1)
        deg = torch.where(deg > 0, deg, torch.ones_like(deg))
        d_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        return d_inv_sqrt @ a @ d_inv_sqrt

    @staticmethod
    def rescale_laplacian(l: torch.Tensor) -> torch.Tensor:
        try:
            eigvals = torch.linalg.eigvals(l).real
            lam = float(eigvals.max().item())
            if lam <= 0:
                lam = 2.0
        except Exception:
            lam = 2.0

        return (2.0 / lam) * l - torch.eye(l.shape[0], device=l.device)


def _build_od_tensors(
    od_df: pd.DataFrame,
    city_ids: List[int],
    dates: List[pd.Timestamp],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    city_to_idx = {cid: i for i, cid in enumerate(city_ids)}
    date_to_idx = {str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)}

    T = len(dates)
    N = len(city_ids)


    od_raw_observed = np.full((T, N, N, 1), np.nan, dtype=np.float32)
    od_obs_mask = np.zeros((T, N, N, 1), dtype=np.float32)

    tmp = od_df.copy()
    tmp["date_key"] = pd.to_datetime(tmp["date"]).dt.date.astype(str)

    for _, row in tmp.iterrows():
        di = date_to_idx[row["date_key"]]
        oi = city_to_idx[int(row["origin_id"])]
        dj = city_to_idx[int(row["destination_id"])]
        od_raw_observed[di, oi, dj, 0] = float(row["od_flow"])
        od_obs_mask[di, oi, dj, 0] = 1.0


    od_raw = np.nan_to_num(od_raw_observed, nan=0.0).astype(np.float32)


    od_raw_for_input = od_raw_observed.copy()
    for oi in range(N):
        for dj in range(N):
            series = pd.Series(od_raw_for_input[:, oi, dj, 0])
            series = series.ffill().bfill().fillna(0.0)
            od_raw_for_input[:, oi, dj, 0] = series.to_numpy(dtype=np.float32)

    od_log = np.log1p(np.clip(od_raw_for_input, a_min=0.0, a_max=None)).astype(np.float32)

    return od_raw, od_log, od_obs_mask

def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if xmax - xmin < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + 1e-8)


def _build_directed_adj(pair_static_df: pd.DataFrame, city_ids: List[int]) -> np.ndarray:
    city_to_idx = {cid: i for i, cid in enumerate(city_ids)}
    n = len(city_ids)
    adj = np.zeros((n, n), dtype=np.float32)
    df = pair_static_df.copy()


    distance = pd.to_numeric(df.get("distance_road", df.get("distance_line", 1.0)), errors="coerce").fillna(1.0).to_numpy(np.float32)
    distance = np.where(distance <= 0, 1.0, distance)


    inv_dist = _normalize_01(1.0 / distance)


    hsr_direct = pd.to_numeric(df.get("hsr_direct_flag", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    hsr_service = pd.to_numeric(df.get("hsr_service_intensity", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    hsr_service = _normalize_01(hsr_service)
    adjacent = pd.to_numeric(df.get("is_adjacent", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
    same_province = pd.to_numeric(df.get("same_province", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)


    weights = 0.30 * inv_dist + 0.25 * hsr_service + 0.20 * hsr_direct + 0.15 * adjacent + 0.10 * same_province

    for row_idx, (_, row) in enumerate(df.iterrows()):
        oi = city_to_idx[int(row["origin_id"])]
        dj = city_to_idx[int(row["destination_id"])]
        adj[oi, dj] = weights[row_idx]


    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)

def _build_pair_static_tensor(pair_static_df: pd.DataFrame, city_ids: List[int]) -> np.ndarray:
    city_to_idx = {cid: i for i, cid in enumerate(city_ids)}
    n = len(city_ids)

    feature_cols = [
        "distance_line",
        "distance_road",
        "distance_railway",
        "is_adjacent",
        "same_province",
        "gdp_gap",
        "income_gap",
        "population_gap",
        "industry_structure_similarity",
        "poi_structure_similarity",
        "hsr_direct_flag",
        "hsr_train_count",
        "hsr_min_travel_time",
        "hsr_avg_travel_time",
        "hsr_service_intensity",
    ]

    binary_cols = {
        "is_adjacent",
        "same_province",
        "hsr_direct_flag",
    }

    df = pair_static_df.copy()
    feat_list = []

    for col in feature_cols:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float32)
        else:
            arr = np.zeros((len(df),), dtype=np.float32)

        if col not in binary_cols:
            arr = _normalize_01(arr)

        feat_list.append(arr)

    feat_mat = np.stack(feat_list, axis=-1).astype(np.float32)
    pair_static_num = np.zeros((n, n, feat_mat.shape[-1]), dtype=np.float32)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        oi = city_to_idx[int(row["origin_id"])]
        dj = city_to_idx[int(row["destination_id"])]
        pair_static_num[oi, dj] = feat_mat[row_idx]

    return pair_static_num.astype(np.float32)

def _build_symmetric_adj(base_adj: np.ndarray) -> np.ndarray:
    sym = 0.5 * (base_adj + base_adj.T)


    np.fill_diagonal(sym, 1.0)
    return sym.astype(np.float32)


def _cyclical_encode(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * values.astype(np.float32) / float(period)
    return np.sin(angle), np.cos(angle)


def _build_regime_tensor(
        city_dynamic_df: pd.DataFrame,
        pair_weather_df: pd.DataFrame,
        dates: List[pd.Timestamp]
    ) -> np.ndarray:
    city_df = city_dynamic_df.copy()
    city_df["date"] = pd.to_datetime(city_df["date"])

    pair_df = pair_weather_df.copy()
    pair_df["date"] = pd.to_datetime(pair_df["date"])


    city_num = city_df.groupby("date").agg(
        is_weekend=("is_weekend", "mean"),
        is_holiday=("is_holiday", "mean"),
        holiday_type=(
            "holiday_type",
            lambda s: (
                s.dropna().astype(str).iloc[0]
                if len(s.dropna()) > 0 else "NONE"
            ),
        ),
        month=("month", "first"),
        day_of_week=("day_of_week", "first"),
        temp_mean=("temp_mean", "mean"),
        wind_speed=("wind_speed", "mean"),
        snow_flag=("snow_flag", "mean"),
        dewpoint_temperature=("dewpoint_temperature", "mean"),
    )


    holiday_type = city_num["holiday_type"].fillna("NONE").astype(str)
    holiday_type = holiday_type.replace({"": "NONE", "nan": "NONE", "None": "NONE"})


    holiday_ohe = pd.get_dummies(holiday_type, prefix="holiday_type", dtype=np.float32)


    pair_num = pair_df.groupby("date").agg(
        temp_diff=("temp_diff", "mean"),
        wind_max=("wind_max", "mean"),
        snow_any=("snow_any", "mean"),
    )


    merged = city_num.drop(columns=["holiday_type"]).join(pair_num, how="left")
    merged = merged.join(holiday_ohe, how="left")
    merged = merged.reindex(pd.to_datetime(dates)).ffill().bfill().fillna(0.0)


    month_sin, month_cos = _cyclical_encode(merged["month"].to_numpy(), 12.0)
    dow_sin, dow_cos = _cyclical_encode(merged["day_of_week"].to_numpy(), 7.0)


    holiday_cols = sorted([c for c in merged.columns if c.startswith("holiday_type_")])

    regime_parts = [
        merged["is_weekend"].to_numpy(np.float32),
        merged["is_holiday"].to_numpy(np.float32),
        month_sin.astype(np.float32),
        month_cos.astype(np.float32),
        dow_sin.astype(np.float32),
        dow_cos.astype(np.float32),
        merged["temp_mean"].to_numpy(np.float32),
        merged["wind_speed"].to_numpy(np.float32),
        merged["snow_flag"].to_numpy(np.float32),
        merged["dewpoint_temperature"].to_numpy(np.float32),
        merged["temp_diff"].to_numpy(np.float32),
        merged["wind_max"].to_numpy(np.float32),
        merged["snow_any"].to_numpy(np.float32),
    ]


    for col in holiday_cols:
        regime_parts.append(merged[col].to_numpy(np.float32))

    regime = np.stack(regime_parts, axis=-1)
    return regime.astype(np.float32)


def _zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def _empty_semantics(n: int) -> Dict[str, np.ndarray]:
    d = {
        "origin_city_static_sem": _zeros((n, 8), np.int64),
        "destination_city_static_sem": _zeros((n, 8), np.int64),
        "pair_relation_static_sem": _zeros((n, n, 7), np.int64),
        "origin_city_static_sem_prob": np.full((n, 8, 5), 0.2, np.float32),
        "destination_city_static_sem_prob": np.full((n, 8, 5), 0.2, np.float32),
        "pair_relation_static_sem_prob": np.full((n, n, 7, 5), 0.2, np.float32),
        "origin_city_static_sem_confidence": np.full((n, 8), 0.5, np.float32),
        "destination_city_static_sem_confidence": np.full((n, 8), 0.5, np.float32),
        "pair_relation_static_sem_confidence": np.full((n, n, 7), 0.5, np.float32),
        "origin_city_static_sem_entropy": np.full((n, 8), 1.0, np.float32),
        "destination_city_static_sem_entropy": np.full((n, 8), 1.0, np.float32),
        "pair_relation_static_sem_entropy": np.full((n, n, 7), 1.0, np.float32),
        "pair_mask": np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32),
    }
    return d


def _resolve_semantic_npz(semantic_path: str | Path | None) -> Optional[Path]:
    if semantic_path is None:
        return None

    p = Path(semantic_path)


    if p.is_file():
        return p


    candidates = [
        p / "role_semantic_tensors.npz",
        p / "tensors" / "role_semantic_tensors.npz",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def _load_semantics(semantic_path: str | Path | None, n: int) -> Dict[str, np.ndarray]:
    resolved = _resolve_semantic_npz(semantic_path)

    if resolved is None or (not resolved.exists()):
        return _empty_semantics(n)

    arr = np.load(resolved)


    out = _empty_semantics(n)

    for k in out.keys():
        if k in arr.files:
            out[k] = arr[k].astype(out[k].dtype, copy=False)

    return out


def load_group_bundle(
    group_dir: str | Path,
    semantic_path: str | Path | None,
    domain_id: int,
    kernel_type: str = "chebyshev",
    K: int = 3,
) -> GroupDataBundle:
    group_dir = Path(group_dir)
    group = group_dir.name


    od_df = pd.read_csv(group_dir / f"{group}_OD.txt", sep="\t")
    city_static_df = pd.read_csv(group_dir / f"{group}_city_static.txt", sep="\t")
    city_dynamic_df = pd.read_csv(group_dir / f"{group}_city_dynamic.txt", sep="\t")
    pair_static_df = pd.read_csv(group_dir / f"{group}_city_pair_static.txt", sep="\t")
    pair_weather_df = pd.read_csv(group_dir / f"{group}_pair_weather.txt", sep="\t")

    od_df["date"] = pd.to_datetime(od_df["date"])


    city_ids = sorted(city_static_df["city_id"].astype(int).tolist())
    dates = sorted(od_df["date"].drop_duplicates().tolist())


    od_raw, od_log, od_obs_mask = _build_od_tensors(od_df, city_ids, dates)

    base_adj = _build_directed_adj(pair_static_df, city_ids)
    sym_adj = _build_symmetric_adj(base_adj)


    pair_static_num = _build_pair_static_tensor(pair_static_df, city_ids)


    graph_kernel = AdjProcessor(kernel_type=kernel_type, K=K).process(sym_adj)

    regime = _build_regime_tensor(city_dynamic_df, pair_weather_df, dates)
    sem = _load_semantics(semantic_path, len(city_ids))

    return GroupDataBundle(
        group_name=group,
        domain_id=domain_id,
        city_ids=city_ids,
        dates=[str(pd.Timestamp(d).date()) for d in dates],
        od_raw=od_raw,
        od_log=od_log,
        od_obs_mask=od_obs_mask,
        base_adj=base_adj,
        sym_adj=sym_adj,
        graph_kernel=graph_kernel,
        regime=regime,
        pair_static_num=pair_static_num,
        node_mask=np.ones((len(city_ids),), dtype=np.float32),
        pair_mask=sem["pair_mask"].astype(np.float32),
        sem=sem,
    )


def pad_bundle(bundle: GroupDataBundle, target_n: int) -> GroupDataBundle:
    cur_n = len(bundle.city_ids)
    if cur_n == target_n:
        return bundle

    def pad_array(arr: np.ndarray, final_shape: Sequence[int]) -> np.ndarray:
        out = np.zeros(final_shape, dtype=arr.dtype)
        slices = tuple(slice(0, s) for s in arr.shape)
        out[slices] = arr
        return out

    t = bundle.od_raw.shape[0]
    new_sem: Dict[str, np.ndarray] = {}


    for k, v in bundle.sem.items():
        if v.ndim == 2 and v.shape[0] == cur_n:
            new_sem[k] = pad_array(v, (target_n, v.shape[1]))
        elif v.ndim == 3 and v.shape[0] == cur_n and v.shape[1] == cur_n:
            new_sem[k] = pad_array(v, (target_n, target_n, v.shape[2]))
        elif v.ndim == 3 and v.shape[0] == cur_n:
            new_sem[k] = pad_array(v, (target_n, v.shape[1], v.shape[2]))
        elif v.ndim == 4 and v.shape[0] == cur_n and v.shape[1] == cur_n:
            new_sem[k] = pad_array(v, (target_n, target_n, v.shape[2], v.shape[3]))
        else:
            new_sem[k] = v.copy()


    pair_mask = np.zeros((target_n, target_n), dtype=np.float32)
    pair_mask[:cur_n, :cur_n] = bundle.pair_mask

    node_mask = np.zeros((target_n,), dtype=np.float32)
    node_mask[:cur_n] = 1.0

    return GroupDataBundle(
        group_name=bundle.group_name,
        domain_id=bundle.domain_id,
        city_ids=bundle.city_ids + [-1] * (target_n - cur_n),
        dates=bundle.dates,
        od_raw=pad_array(bundle.od_raw, (t, target_n, target_n, 1)),
        od_log=pad_array(bundle.od_log, (t, target_n, target_n, 1)),
        od_obs_mask=pad_array(bundle.od_obs_mask, (t, target_n, target_n, 1)),
        base_adj=pad_array(bundle.base_adj, (target_n, target_n)),
        sym_adj=pad_array(bundle.sym_adj, (target_n, target_n)),
        graph_kernel=pad_array(bundle.graph_kernel, (bundle.graph_kernel.shape[0], target_n, target_n)),
        regime=bundle.regime.copy(),
        pair_static_num=pad_array(
            bundle.pair_static_num,
            (target_n, target_n, bundle.pair_static_num.shape[-1]),
        ),
        node_mask=node_mask,
        pair_mask=pair_mask,
        sem=new_sem,
    )

class MultiCityWindowDataset(Dataset):

    def __init__(self, bundles: List[GroupDataBundle], samples: List[Tuple[int, int]], hist_len: int, pred_len: int):
        self.bundles = bundles
        self.samples = samples
        self.hist_len = hist_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        bundle_idx, t = self.samples[idx]
        b = self.bundles[bundle_idx]
        sem = b.sem

        item: Dict[str, torch.Tensor | str] = {
            "group_name": b.group_name,
            "domain_id": torch.tensor(b.domain_id).long(),
            "x_od": torch.from_numpy(b.od_log[t - self.hist_len:t]).float(),
            "y_od": torch.from_numpy(b.od_raw[t:t + self.pred_len]).float(),
            "y_obs_mask": torch.from_numpy(b.od_obs_mask[t:t + self.pred_len]).float(),
            "x_regime": torch.from_numpy(b.regime[t - self.hist_len:t]).float(),
            "y_regime": torch.from_numpy(b.regime[t:t + self.pred_len]).float(),
            "graph_kernel": torch.from_numpy(b.graph_kernel).float(),
            "base_adj": torch.from_numpy(b.base_adj).float(),
            "pair_static_num": torch.from_numpy(b.pair_static_num).float(),
            "pair_mask": torch.from_numpy(b.pair_mask).float(),
            "node_mask": torch.from_numpy(b.node_mask).float(),
        }


        static_keys = [
            "origin_city_static_sem",
            "destination_city_static_sem",
            "pair_relation_static_sem",
            "origin_city_static_sem_prob",
            "destination_city_static_sem_prob",
            "pair_relation_static_sem_prob",
            "origin_city_static_sem_confidence",
            "destination_city_static_sem_confidence",
            "pair_relation_static_sem_confidence",
            "origin_city_static_sem_entropy",
            "destination_city_static_sem_entropy",
            "pair_relation_static_sem_entropy",
        ]

        for key in static_keys:
            item[key] = torch.from_numpy(sem[key])

        return item


def _time_windows(total_t: int, hist_len: int, pred_len: int) -> np.ndarray:
    return np.arange(hist_len, total_t - pred_len + 1)

def _valid_time_windows_from_dates(
    dates,
    hist_len: int,
    pred_len: int,
) -> np.ndarray:
    date_arr = pd.to_datetime(list(dates)).to_numpy(dtype="datetime64[D]")
    all_t = np.arange(hist_len, len(date_arr) - pred_len + 1, dtype=np.int64)

    keep = []
    for t in all_t:
        span = date_arr[t - hist_len:t + pred_len]
        gap = np.diff(span).astype("timedelta64[D]").astype(np.int64)
        if np.all(gap == 1):
            keep.append(int(t))

    return np.asarray(keep, dtype=np.int64)

def _split_windows(indices: np.ndarray, ratios: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    n = len(indices)
    total = float(sum(ratios))
    n_train = int(n * ratios[0] / total)
    n_val = int(n * ratios[1] / total)

    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def build_experiment_data(config: Dict) -> Tuple[List[GroupDataBundle], Dict[str, DataLoader], Dict]:

    group_dirs = [Path(x) for x in config["group_dirs"]]
    semantic_paths = config.get("semantic_npzs") or [None] * len(group_dirs)

    assert len(group_dirs) == len(semantic_paths), "group_dirs and semantic_npzs must have the same length."

    kernel_type = config.get("kernel_type", "chebyshev")
    K = int(config.get("K", 3))


    bundles = [
        load_group_bundle(group_dir=g, semantic_path=s, domain_id=i, kernel_type=kernel_type, K=K)
        for i, (g, s) in enumerate(zip(group_dirs, semantic_paths))
    ]


    max_nodes = max(len(b.city_ids) for b in bundles)
    bundles = [pad_bundle(b, max_nodes) for b in bundles]

    hist_len = int(config["num_history"])
    pred_len = int(config["future_steps"])
    split_ratio = tuple(config.get("split_ratio", (7, 1, 2)))
    task_type = config.get("task_type", "single_city")
    target_group = config.get("target_group")
    few_shot_ratio = float(config.get("few_shot_ratio", 0.1))
    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))
    max_train_windows = config.get("max_train_windows")
    max_val_windows = config.get("max_val_windows")
    max_test_windows = config.get("max_test_windows")

    split_samples: Dict[str, List[Tuple[int, int]]] = {"train": [], "val": [], "test": []}
    extra = {}

    if task_type == "single_city":

        assert len(bundles) == 1, "single_city mode expects exactly one group."
        idxs = _valid_time_windows_from_dates(bundles[0].dates, hist_len, pred_len)
        parts = _split_windows(idxs, split_ratio)
        for split, arr in parts.items():
            split_samples[split] = [(0, int(t)) for t in arr]

    elif task_type == "multicity_joint":

        for bi, b in enumerate(bundles):
            idxs = _valid_time_windows_from_dates(b.dates, hist_len, pred_len)
            parts = _split_windows(idxs, split_ratio)
            for split, arr in parts.items():
                split_samples[split].extend((bi, int(t)) for t in arr)

    elif task_type == "leave_one_city_out":

        assert target_group is not None, "leave_one_city_out requires target_group."
        target_idx = next(i for i, b in enumerate(bundles) if b.group_name == target_group)

        for bi, b in enumerate(bundles):
            idxs = _valid_time_windows_from_dates(b.dates, hist_len, pred_len)
            parts = _split_windows(idxs, split_ratio)

            if bi == target_idx:
                split_samples["test"].extend((bi, int(t)) for t in idxs)
            else:
                split_samples["train"].extend((bi, int(t)) for t in parts["train"])
                split_samples["val"].extend((bi, int(t)) for t in parts["val"])

        extra["target_group"] = target_group

    elif task_type == "few_shot_target":


        assert target_group is not None, "few_shot_target requires target_group."
        target_idx = next(i for i, b in enumerate(bundles) if b.group_name == target_group)

        for bi, b in enumerate(bundles):
            idxs = _valid_time_windows_from_dates(b.dates, hist_len, pred_len)
            parts = _split_windows(idxs, split_ratio)

            if bi != target_idx:
                split_samples["train"].extend((bi, int(t)) for t in parts["train"])
                split_samples["val"].extend((bi, int(t)) for t in parts["val"])
            else:
                n_support = max(1, int(len(parts["train"]) * few_shot_ratio))
                support = parts["train"][:n_support]
                remain = parts["train"][n_support:]

                split_samples["train"].extend((bi, int(t)) for t in support)
                split_samples["val"].extend((bi, int(t)) for t in parts["val"])
                split_samples["test"].extend((bi, int(t)) for t in np.concatenate([remain, parts["test"]]))

        extra["target_group"] = target_group
        extra["few_shot_ratio"] = few_shot_ratio

    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Valid options are 'single_city', 'multicity_joint', 'leave_one_city_out', and 'few_shot_target'")


    if max_train_windows is not None:
        split_samples["train"] = split_samples["train"][: int(max_train_windows)]
    if max_val_windows is not None:
        split_samples["val"] = split_samples["val"][: int(max_val_windows)]
    if max_test_windows is not None:
        split_samples["test"] = split_samples["test"][: int(max_test_windows)]


    datasets = {k: MultiCityWindowDataset(bundles, v, hist_len, pred_len) for k, v in split_samples.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


    meta = {
        "max_nodes": max_nodes,
        "num_domains": len(bundles),
        "group_names": [b.group_name for b in bundles],
        "regime_dim": int(bundles[0].regime.shape[-1]),
        "origin_sem_dim": int(bundles[0].sem["origin_city_static_sem"].shape[-1]),
        "destination_sem_dim": int(bundles[0].sem["destination_city_static_sem"].shape[-1]),
        "pair_sem_dim": int(bundles[0].sem["pair_relation_static_sem"].shape[-1]),
        "pair_static_dim": int(bundles[0].pair_static_num.shape[-1]),
    }
    meta.update(extra)

    return bundles, loaders, meta
