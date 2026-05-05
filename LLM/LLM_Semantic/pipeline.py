from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .api_client import DeepSeekClient
from .dataset_processor import (
    build_dense_outputs,
    load_group_dataset,
)
from .io_utils import (
    append_jsonl_row,
    dump_json,
    dump_json_atomic,
    ensure_dir,
    load_json,
    load_jsonl_as_dict,
    normalized_entropy,
    stable_hash,
)
from .prompts import (
    destination_city_static_prompts,
    origin_city_static_prompts,
    pair_relation_static_prompts,
)
from .schemas import stage_keys


@dataclass
class PipelineArgs:

    data_dir: str
    output_dir: str
    config_path: str
    groups: List[str]
    skip_existing: bool = True


class DualSemanticPipeline:

    def __init__(self, args: PipelineArgs):
        self.args = args
        self.output_dir = ensure_dir(args.output_dir)
        self.client = DeepSeekClient(args.config_path)


    def run(self) -> None:
        for group_name in self.args.groups:
            print(f"\n========== Processing group: {group_name} ==========")
            self.run_group(group_name)

    def run_group(self, group_name: str) -> None:

        group_dir = Path(self.args.data_dir) / group_name
        group_out = ensure_dir(self.output_dir / group_name)
        cache_dir = ensure_dir(group_out / "cache")
        table_dir = ensure_dir(group_out / "tables")
        tensor_dir = ensure_dir(group_out / "tensors")

        group_data = load_group_dataset(group_dir)


        origin_city_static_sem = self._process_table(
            stage_name="origin_city_static",
            df=group_data.city_static,
            prompt_builder=origin_city_static_prompts,
            cache_path=cache_dir / "origin_city_static.jsonl",
            cache_key_builder=lambda row: f"origin_city_static::{group_name}::{int(row['city_id'])}",
        )
        origin_city_static_sem.to_csv(
            table_dir / "origin_city_static_semantics.txt",
            sep="	",
            index=False,
            encoding="utf-8",
        )

        destination_city_static_sem = self._process_table(
            stage_name="destination_city_static",
            df=group_data.city_static,
            prompt_builder=destination_city_static_prompts,
            cache_path=cache_dir / "destination_city_static.jsonl",
            cache_key_builder=lambda row: f"destination_city_static::{group_name}::{int(row['city_id'])}",
        )

        destination_city_static_sem.to_csv(
            table_dir / "destination_city_static_semantics.txt",
            sep="	",
            index=False,
            encoding="utf-8",
        )


        pair_relation_static_sem = self._process_table(
            stage_name="pair_relation_static",
            df=group_data.pair_static,
            prompt_builder=pair_relation_static_prompts,
            cache_path=cache_dir / "pair_relation_static.jsonl",
            cache_key_builder=lambda row: f"pair_relation_static::{group_name}::{int(row['origin_id'])}->{int(row['destination_id'])}",
        )
        pair_relation_static_sem.to_csv(
            table_dir / "pair_relation_static_semantics.txt",
            sep="	",
            index=False,
            encoding="utf-8",
        )

        arrays, meta = build_dense_outputs(
            group_data=group_data,
            origin_city_static_sem=origin_city_static_sem,
            destination_city_static_sem=destination_city_static_sem,
            pair_relation_static_sem=pair_relation_static_sem,
        )

        np.savez_compressed(tensor_dir / "role_semantic_tensors.npz", **arrays)
        dump_json(meta, tensor_dir / "role_semantic_meta.json")

        print(f"Finished {group_name}; outputs saved to: {group_out}")

    def _partial_cache_path(self, cache_path: Path, cache_key: str) -> Path:
        partial_dir = ensure_dir(cache_path.parent / f"{cache_path.stem}_partials")
        return partial_dir / f"{stable_hash(cache_key)}.json"

    def _load_partial_samples(
        self,
        partial_path: Path,
        stage_name: str,
        cache_key: str,
        prompt_hash: str,
        temperatures: List[float],
    ) -> List[Dict]:
        if not partial_path.exists():
            return []

        try:
            payload = load_json(partial_path)
        except Exception:
            return []


        same_plan = payload.get("temperature_schedule") == temperatures

        if payload.get("stage_name") != stage_name:
            return []
        if payload.get("cache_key") != cache_key:
            return []
        if payload.get("prompt_hash") != prompt_hash:
            return []
        if not same_plan:
            return []

        raw_samples = payload.get("raw_samples", [])
        if not isinstance(raw_samples, list):
            return []
        return raw_samples[: len(temperatures)]

    def _query_with_self_consistency(
        self,
        stage_name: str,
        system_prompt: str,
        user_prompt: str,
        cache_path: Path,
        cache_key: str,
    ) -> Tuple[Dict, List[Dict], List[float]]:
        temperatures = self.client.build_sampling_plan(stage_name)
        prompt_hash = stable_hash(
            "||".join(
                [
                    stage_name,
                    system_prompt,
                    user_prompt,
                    ",".join(f"{t:.6f}" for t in temperatures),
                ]
            )
        )
        partial_path = self._partial_cache_path(cache_path, cache_key)
        samples: List[Dict] = self._load_partial_samples(
            partial_path=partial_path,
            stage_name=stage_name,
            cache_key=cache_key,
            prompt_hash=prompt_hash,
            temperatures=temperatures,
        )

        start_idx = len(samples)
        for temp in temperatures[start_idx:]:
            sample = self.client.chat_json(
                stage=stage_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
            )
            samples.append(sample)


            dump_json_atomic(
                {
                    "cache_key": cache_key,
                    "stage_name": stage_name,
                    "prompt_hash": prompt_hash,
                    "temperature_schedule": temperatures,
                    "completed_samples": len(samples),
                    "raw_samples": samples,
                },
                partial_path,
            )

        aggregated = self._aggregate_samples(stage_name, samples, temperatures)
        if partial_path.exists():
            partial_path.unlink(missing_ok=True)
        return aggregated, samples, temperatures

    def _aggregate_samples(self, stage_name: str, samples: List[Dict], temperatures: List[float]) -> Dict:
        semantic_keys = stage_keys(stage_name)
        result: Dict[str, object] = {
            "sample_count": int(len(samples)),
            "temperature_schedule": ",".join(f"{t:.2f}" for t in temperatures),
        }


        for key in semantic_keys:
            weighted_votes = np.zeros(5, dtype=np.float64)
            score_sum = 0.0
            conf_sum = 0.0
            rationale_pool: List[Tuple[float, str, int]] = []

            for sample in samples:
                score = int(sample.get(key, 0))
                score = max(0, min(4, score))
                confidence = float(sample.get(f"{key}_confidence", 0.50))
                confidence = max(0.0, min(1.0, confidence))
                rationale = str(sample.get(f"{key}_rationale", "")).strip()


                weight = max(confidence, 0.05)
                weighted_votes[score] += weight
                score_sum += score * weight
                conf_sum += weight
                rationale_pool.append((weight, rationale, score))

            if conf_sum <= 0:
                probs = np.ones(5, dtype=np.float64) / 5.0
            else:
                probs = weighted_votes / conf_sum


            hard_label = int(np.argmax(probs))
            aggregated_confidence = float(probs[hard_label])
            entropy = float(normalized_entropy(probs.tolist()))
            mean_score = float(score_sum / conf_sum) if conf_sum > 0 else 0.0


            chosen_rationale = ""
            ranked_pool = sorted(rationale_pool, key=lambda x: x[0], reverse=True)
            for weight, rationale, score in ranked_pool:
                if score == hard_label and rationale:
                    chosen_rationale = rationale
                    break
            if not chosen_rationale:
                for weight, rationale, score in ranked_pool:
                    if rationale:
                        chosen_rationale = rationale
                        break

            result[key] = hard_label
            result[f"{key}_confidence"] = round(aggregated_confidence, 6)
            result[f"{key}_entropy"] = round(entropy, 6)
            result[f"{key}_mean_score"] = round(mean_score, 6)
            result[f"{key}_rationale"] = chosen_rationale[:60]
            for level in range(5):
                result[f"{key}_prob_{level}"] = round(float(probs[level]), 6)


        summary_candidates: List[Tuple[float, str]] = []
        for sample in samples:
            confs = [float(sample.get(f"{key}_confidence", 0.5)) for key in semantic_keys]
            weight = float(np.mean(confs)) if confs else 0.5
            summary = str(sample.get("summary", "")).replace("\n", " ").strip()
            if summary:
                summary_candidates.append((weight, summary))

        if summary_candidates:
            summary_candidates.sort(key=lambda x: x[0], reverse=True)
            result["summary"] = summary_candidates[0][1][:160]
        else:
            result["summary"] = ""

        return result

    def _process_table(
        self,
        stage_name: str,
        df: pd.DataFrame,
        prompt_builder: Callable[[Dict], Tuple[str, str]],
        cache_path: Path,
        cache_key_builder: Callable[[pd.Series], str],
    ) -> pd.DataFrame:
        cache = load_jsonl_as_dict(cache_path)
        rows = []
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"{stage_name} rows")

        for _, row in iterator:
            row_dict = row.to_dict()
            cache_key = cache_key_builder(row)
            cache_row = cache.get(cache_key, {})


            if self.args.skip_existing and "result" in cache_row:
                result = cache_row["result"]
            else:
                system_prompt, user_prompt = prompt_builder(row_dict)
                result, raw_samples, temperatures = self._query_with_self_consistency(
                    stage_name,
                    system_prompt,
                    user_prompt,
                    cache_path=cache_path,
                    cache_key=cache_key,
                )
                cache_entry = {
                    "cache_key": cache_key,
                    "result": result,
                    "raw_samples": raw_samples,
                    "temperature_schedule": temperatures,
                }
                append_jsonl_row(cache_path, cache_entry)
                cache[cache_key] = cache_entry

            rows.append({**row_dict, **result})

        return pd.DataFrame(rows)
