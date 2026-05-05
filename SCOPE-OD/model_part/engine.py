from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .data import build_experiment_data
from .losses import build_uncertainty_weight, flow_loss, semantic_alignment_loss
from .metrics import masked_metrics_np
from .model import ScopeODNet
from .paras import Args, list_of_param_dicts
from .utils import detach_to_numpy, ensure_dir, load_json, merge_dict_of_scalars, normalize_metrics, save_json, set_seed, to_device


def build_run_name(para: Args) -> str:
    groups = "-".join(Path(x).name for x in para.group_dirs)
    target = f"_tgt{para.target_group}" if getattr(para, "target_group", None) else ""
    return f"{para.task_type}_{groups}{target}_his{para.num_history}_pred{para.future_steps}_hid{para.hidden_dim}_seed{para.seed}"


def _is_dir_empty(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and not any(path.iterdir())


def allocate_run_dir(save_root: str | Path) -> Path:
    save_root = ensure_dir(save_root)

    existing_run_dirs = []

    for child in save_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("Run_"):
            continue

        suffix = child.name[4:]

        if suffix.isdigit():
            existing_run_dirs.append((int(suffix), child))

    for _, run_dir in sorted(existing_run_dirs, key=lambda x: x[0]):
        if _is_dir_empty(run_dir):
            return run_dir

    next_idx = 1 if not existing_run_dirs else max(idx for idx, _ in existing_run_dirs) + 1
    run_dir = save_root / f"Run_{next_idx}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_existing_all_results(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    try:
        data = load_json(path)
    except Exception as exc:
        print(f"[warning] Failed to read {path}; starting from an empty result list. Error: {exc}")
        return []

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]

    if isinstance(data, list):
        return data

    print(f"[warning] Unexpected format in {path}; starting from an empty result list.")
    return []


def maybe_freeze_for_lora(model: torch.nn.Module, para: Args) -> None:
    if not getattr(para, "use_lora_adaptation", False):
        return
    if not getattr(para, "freeze_base_when_lora", False):
        return
    if para.task_type != "few_shot_target":
        return

    for name, p in model.named_parameters():
        if ("lora_" in name) or ("output_head" in name) or ("sem_head" in name):

            p.requires_grad = True
        else:


            if p.requires_grad:
                p.requires_grad = False


def _pair_flow_mask(batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
    pair_mask = batch["pair_mask"].unsqueeze(1).unsqueeze(-1)
    pair_mask = pair_mask.expand(-1, horizon, -1, -1, 1)


    if "y_obs_mask" in batch:
        pair_mask = pair_mask * batch["y_obs_mask"].float()

    return pair_mask


def compute_total_loss(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], para: Args) -> Dict[str, torch.Tensor]:

    mask = _pair_flow_mask(batch, batch["y_od"].shape[1])


    flow = flow_loss(
        outputs["pred_raw"],
        batch["y_od"],
        mask=mask,
        huber_delta=float(getattr(para, "huber_delta", 1.0)),
        wmape_weight=float(getattr(para, "wmape_weight", 0.2)),
    )


    if getattr(para, "use_semantic_alignment", True):
        if getattr(para, "use_uncertainty_weighting", False):

            origin_weight = build_uncertainty_weight(
                batch["origin_city_static_sem_prob"],
                batch.get("origin_city_static_sem_confidence"),
                batch.get("origin_city_static_sem_entropy"),
                mode=getattr(para, "uncertainty_mode", "prob_margin"),
                min_weight=float(getattr(para, "uncertainty_min_weight", 0.3)),
            )
            destination_weight = build_uncertainty_weight(
                batch["destination_city_static_sem_prob"],
                batch.get("destination_city_static_sem_confidence"),
                batch.get("destination_city_static_sem_entropy"),
                mode=getattr(para, "uncertainty_mode", "prob_margin"),
                min_weight=float(getattr(para, "uncertainty_min_weight", 0.3)),
            )
            pair_weight = build_uncertainty_weight(
                batch["pair_relation_static_sem_prob"],
                batch.get("pair_relation_static_sem_confidence"),
                batch.get("pair_relation_static_sem_entropy"),
                mode=getattr(para, "uncertainty_mode", "prob_margin"),
                min_weight=float(getattr(para, "uncertainty_min_weight", 0.3)),
            )
        else:
            origin_weight = destination_weight = pair_weight = None


        sem_origin = semantic_alignment_loss(
            outputs["origin_static_logits"],
            batch["origin_city_static_sem"],
            batch["origin_city_static_sem_prob"],
            mask=batch["node_mask"].unsqueeze(-1).expand_as(batch["origin_city_static_sem"]).float(),
            sample_weight=origin_weight,
            lambda_ce=float(getattr(para, "lambda_sem_ce", 0.0)),
            lambda_kl=1.0,
        )


        sem_destination = semantic_alignment_loss(
            outputs["destination_static_logits"],
            batch["destination_city_static_sem"],
            batch["destination_city_static_sem_prob"],
            mask=batch["node_mask"].unsqueeze(-1).expand_as(batch["destination_city_static_sem"]).float(),
            sample_weight=destination_weight,
            lambda_ce=float(getattr(para, "lambda_sem_ce", 0.0)),
            lambda_kl=1.0,
        )


        sem_pair = semantic_alignment_loss(
            outputs["pair_static_logits"],
            batch["pair_relation_static_sem"],
            batch["pair_relation_static_sem_prob"],
            mask=batch["pair_mask"].unsqueeze(-1).expand_as(batch["pair_relation_static_sem"]).float(),
            sample_weight=pair_weight,
            lambda_ce=float(getattr(para, "lambda_sem_ce", 0.0)),
            lambda_kl=1.0,
        )
    else:

        device = outputs["pred_raw"].device
        zero = torch.zeros((), device=device)
        sem_origin = {"total": zero, "ce": zero, "kl": zero}
        sem_destination = {"total": zero, "ce": zero, "kl": zero}
        sem_pair = {"total": zero, "ce": zero, "kl": zero}


    total = (
        flow["total"]
        + float(getattr(para, "lambda_sem_origin", 0.02)) * sem_origin["total"]
        + float(getattr(para, "lambda_sem_destination", 0.02)) * sem_destination["total"]
        + float(getattr(para, "lambda_sem_pair", 0.05)) * sem_pair["total"]
    )

    return {
        "total": total,
        "flow_total": flow["total"],
        "flow_huber_log": flow["huber_log"],
        "flow_wmape": flow["wmape"],
        "flow_mae_raw": flow["mae_raw"],
        "sem_origin": sem_origin["total"],
        "sem_destination": sem_destination["total"],
        "sem_pair": sem_pair["total"],
        "sem_origin_kl": sem_origin["kl"],
        "sem_destination_kl": sem_destination["kl"],
        "sem_pair_kl": sem_pair["kl"],
        "sem_origin_ce": sem_origin["ce"],
        "sem_destination_ce": sem_destination["ce"],
        "sem_pair_ce": sem_pair["ce"],
    }


def run_epoch(model, loader, optimizer, device, para: Args) -> Dict[str, float]:
    model.train()
    agg, n = {}, 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = to_device(batch, device)

        optimizer.zero_grad()
        outputs = model(batch)
        losses = compute_total_loss(batch, outputs, para)


        losses["total"].backward()


        if getattr(para, "grad_clip", 0.0) and para.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), para.grad_clip)

        optimizer.step()


        bs = batch["x_od"].shape[0]
        n += bs
        merge_dict_of_scalars(agg, {k: float(v.item()) for k, v in losses.items()}, weight=bs)

    return normalize_metrics(agg, n)


@torch.no_grad()
def evaluate(model, loader, device, para: Args) -> Dict[str, float]:
    model.eval()
    agg, n = {}, 0
    preds, trues, masks = [], [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = to_device(batch, device)
        outputs = model(batch)
        losses = compute_total_loss(batch, outputs, para)

        bs = batch["x_od"].shape[0]
        n += bs
        merge_dict_of_scalars(agg, {k: float(v.item()) for k, v in losses.items()}, weight=bs)


        preds.append(detach_to_numpy(outputs["pred_raw"]))
        trues.append(detach_to_numpy(batch["y_od"]))
        masks.append(detach_to_numpy(_pair_flow_mask(batch, batch["y_od"].shape[1])))

    agg = normalize_metrics(agg, n)

    if preds:
        pred_np = np.concatenate(preds, axis=0)
        true_np = np.concatenate(trues, axis=0)
        mask_np = np.concatenate(masks, axis=0)


        agg.update(masked_metrics_np(true_np, pred_np, mask=mask_np))

    return agg


def train_one_setting(para: Args) -> Dict[str, object]:

    set_seed(int(para.seed))
    device = torch.device(para.device)


    bundles, loaders, meta = build_experiment_data(para.to_dict())


    save_dir = ensure_dir(
        Path(getattr(para, "run_dir"))
        if getattr(para, "run_dir", None)
        else Path(para.save_root) / build_run_name(para)
    )
    save_json(para.to_dict(), save_dir / "config.json")


    model = ScopeODNet(
        num_nodes=meta["max_nodes"],
        K=int(para.K),
        input_dim=1,
        hidden_dim=int(para.hidden_dim),
        out_horizon=int(para.future_steps),
        num_layers=int(para.num_layers),
        origin_sem_dim=int(meta["origin_sem_dim"]),
        destination_sem_dim=int(meta["destination_sem_dim"]),
        pair_sem_dim=int(meta["pair_sem_dim"]),
        pair_static_dim=int(meta["pair_static_dim"]),
        regime_dim=int(meta["regime_dim"]),
        semantic_hidden_dim=int(para.semantic_hidden_dim),
        semantic_cond_dim=int(para.semantic_cond_dim),
        semantic_vocab_size=int(para.semantic_vocab_size),
        use_regime_token=bool(para.use_regime_token),
        use_film_controller=bool(para.use_film_controller),
        use_pair_gate=bool(para.use_pair_gate),
        use_lora_adaptation=bool(getattr(para, "use_lora_adaptation", False)),
        lora_rank=int(getattr(para, "lora_rank", 4)),
        lora_alpha=float(getattr(para, "lora_alpha", 8.0)),
        freeze_base_when_lora=bool(getattr(para, "freeze_base_when_lora", False)),
    ).to(device)


    maybe_freeze_for_lora(model, para)


    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=para.lr, weight_decay=para.weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(para.num_epochs), 1)
    ) if getattr(para, "use_scheduler", True) else None

    best_val = float("inf")
    best_metrics = None
    history: List[Dict[str, float]] = []


    for epoch in range(1, int(para.num_epochs) + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, device, para)


        val_metrics = evaluate(model, loaders["val"], device, para) if len(loaders["val"].dataset) > 0 else {"total": train_metrics["total"]}

        if scheduler is not None:
            scheduler.step()


        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        print(
            f"[epoch {epoch:03d}] "
            f"train_total={train_metrics.get('total', 0):.6f} "
            f"val_total={val_metrics.get('total', 0):.6f} "
            f"val_MAE={val_metrics.get('MAE', 0):.6f} "
            f"val_MAPE={val_metrics.get('MAPE', 0):.6f} "
            f"val_WMAPE={val_metrics.get('flow_wmape', 0):.6f}"
        )


        best_metric_name = str(getattr(para, "best_metric", "flow_wmape"))
        score = val_metrics.get(best_metric_name, val_metrics.get("flow_wmape", val_metrics.get("MAE", 1e9)))

        if score < best_val:
            best_val = score
            best_metrics = {
                "epoch": epoch,
                "metric_name": best_metric_name,
                "score": float(score),
                **val_metrics,
            }

            torch.save(
                {"model_state": model.state_dict(), "args": para.to_dict(), "meta": meta},
                save_dir / "best_model.pt"
            )


    checkpoint = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, loaders["test"], device, para) if len(loaders["test"].dataset) > 0 else {}


    if len(loaders["test"].dataset) > 0:
        model.eval()
        all_pred, all_true, all_mask = [], [], []

        with torch.no_grad():
            for batch in loaders["test"]:
                batch = to_device(batch, device)
                outputs = model(batch)
                all_pred.append(detach_to_numpy(outputs["pred_raw"]))
                all_true.append(detach_to_numpy(batch["y_od"]))
                all_mask.append(detach_to_numpy(_pair_flow_mask(batch, batch["y_od"].shape[1])))

        np.savez_compressed(
            save_dir / "test_predictions.npz",
            y_pred=np.concatenate(all_pred, axis=0),
            y_true=np.concatenate(all_true, axis=0),
            y_mask=np.concatenate(all_mask, axis=0),
        )


    save_json(history, save_dir / "history.json")
    result = {
        "best_val": best_metrics,
        "test": test_metrics,
        "meta": meta,
        "args": para.to_dict(),
        "run_dir": str(save_dir),
        "run_name": build_run_name(para),
    }
    save_json(result, save_dir / "metrics.json")

    return result


def run_experiments(param_dict: Dict) -> None:
    all_settings = list_of_param_dicts(param_dict)

    print("=" * 100)
    print(f"Detected {len(all_settings)} parameter setting(s).")


    save_root = ensure_dir(
        param_dict["save_root"][0]
        if isinstance(param_dict["save_root"], list)
        else param_dict["save_root"]
    )
    all_results_path = Path(save_root) / "all_results.json"
    all_results = load_existing_all_results(all_results_path)

    if all_results:
        print(f"Found {len(all_results)} existing result(s); appending new runs.")

    for run_idx, parameters in enumerate(all_settings, start=1):
        print("=" * 100)

        current_parameters = dict(parameters)

        run_dir = allocate_run_dir(save_root)
        current_parameters["run_dir"] = str(run_dir)

        print(f"Starting run {run_idx}/{len(all_settings)}")
        print(f"Output directory: {run_dir}\n")

        para = Args(current_parameters)
        result = train_one_setting(para)
        all_results.append(result)


        save_json({"results": all_results}, all_results_path)


    save_json({"results": all_results}, all_results_path)
