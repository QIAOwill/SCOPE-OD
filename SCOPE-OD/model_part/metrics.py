from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _safe_divide(a, b):
    return a / np.clip(b, 1e-8, None)


def masked_metrics_np(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:

    if mask is None:
        mask = np.ones_like(y_true, dtype=np.float32)

    mask = mask.astype(np.float32)


    denom = max(float(mask.sum()), 1.0)


    err = (y_pred - y_true) * mask
    abs_err = np.abs(err)
    sq_err = err ** 2


    mae = abs_err.sum() / denom


    rmse = float(np.sqrt(sq_err.sum() / denom))


    mape = float((_safe_divide(abs_err, np.abs(y_true) + 1.0) * mask).sum() / denom)
    smape = float((2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + 1.0)).sum() / denom)


    wmape = float(abs_err.sum() / max(float((np.abs(y_true) * mask).sum()), 1.0))

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape,
        "WMAPE": wmape
    }


def masked_metrics_torch(y_true: torch.Tensor, y_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

    if mask is None:
        mask = torch.ones_like(y_true)

    denom = mask.sum().clamp_min(1.0)


    err = (y_pred - y_true) * mask
    abs_err = err.abs()
    sq_err = err.square()

    mae = abs_err.sum() / denom
    rmse = torch.sqrt(sq_err.sum() / denom)
    wmape = abs_err.sum() / max(float((torch.abs(y_true) * mask).sum()), 1.0)

    return {"mae": mae, "rmse": rmse, "wmape": wmape}
