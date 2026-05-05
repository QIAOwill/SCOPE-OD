from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def build_uncertainty_weight(
    prob: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    mode: str = "prob_margin",
    min_weight: float = 0.3,
) -> torch.Tensor:

    if mode == "confidence_entropy" and conf is not None and entropy is not None:
        score = conf.float() * (1.0 - entropy.float())
    else:


        top2 = torch.topk(prob.float(), k=2, dim=-1).values
        score = (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)


    return min_weight + (1.0 - min_weight) * score


def flow_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
    wmape_weight: float = 0.2,
    flow_loss_alpha: float = 0.7,
) -> Dict[str, torch.Tensor]:
    if mask is None:
        mask = torch.ones_like(target_raw)

    mask = mask.float()
    pred_raw = pred_raw.clamp_min(0.0)
    target_raw = target_raw.clamp_min(0.0)

    denom = mask.sum().clamp_min(1.0)

    pred_log = torch.log1p(pred_raw)
    target_log = torch.log1p(target_raw)

    huber = F.huber_loss(pred_log, target_log, reduction="none", delta=huber_delta)
    huber = (huber * mask).sum() / denom

    abs_err = (pred_raw - target_raw).abs() * mask
    mae = abs_err.sum() / denom


    wmape = abs_err.sum() / ((target_raw.abs() * mask).sum().clamp_min(1.0))

    total = huber + float(wmape_weight) * wmape

    return {
        "total": total,
        "huber_log": huber,
        "wmape": wmape,
        "mae_raw": mae,
    }


def semantic_alignment_loss(
    logits: torch.Tensor,
    target_hard: torch.Tensor,
    target_prob: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    lambda_ce: float = 0.0,
    lambda_kl: float = 1.0,
) -> Dict[str, torch.Tensor]:

    log_prob = F.log_softmax(logits, dim=-1)


    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_hard.long().reshape(-1),
        reduction="none",
    ).reshape(target_hard.shape)


    kl = F.kl_div(log_prob, target_prob.float(), reduction="none").sum(dim=-1)


    if sample_weight is not None:
        ce = ce * sample_weight
        kl = kl * sample_weight


    if mask is not None:
        ce = ce * mask
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0)
    else:

        denom = torch.tensor(float(target_hard.numel()), device=target_hard.device).clamp_min(1.0)


    ce = ce.sum() / denom
    kl = kl.sum() / denom


    total = lambda_ce * ce + lambda_kl * kl

    return {"total": total, "ce": ce, "kl": kl}
