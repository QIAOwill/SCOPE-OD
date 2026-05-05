from __future__ import annotations

import random
import time
from typing import Any, Dict, List

import requests

from .io_utils import extract_first_json, load_json, stable_hash
from .schemas import sanitize_stage_result, stage_keys


class DeepSeekClient:

    def __init__(self, config_path: str):
        self.config = load_json(config_path)
        self.api_key = self.config.get("api_key", "")
        self.base_url = self.config.get("base_url", "https://api.deepseek.com/v1/chat/completions")
        self.model = self.config.get("model", "deepseek-chat")
        self.timeout = int(self.config.get("timeout", 120))
        self.max_tokens = int(self.config.get("max_tokens", 1400))


        self.temperature = float(self.config.get("temperature", 0.0))

        self.use_json_mode = bool(self.config.get("use_json_mode", True))
        self.sleep_seconds = float(self.config.get("sleep_seconds", 0.2))
        self.max_retries = int(self.config.get("max_retries", 4))
        self.mock_mode = bool(self.config.get("mock_mode", False))


        self.semantic_num_samples = int(self.config.get("semantic_num_samples", 3))
        self.semantic_temperature_list = self.config.get("semantic_temperature_list", [0.0, 0.2, 0.4])
        if not isinstance(self.semantic_temperature_list, list) or not self.semantic_temperature_list:
            self.semantic_temperature_list = [self.temperature]
        self.semantic_temperature_list = [float(x) for x in self.semantic_temperature_list]


        self.stage_num_samples = self.config.get("stage_num_samples", {})
        self.stage_temperature_list = self.config.get("stage_temperature_list", {})

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        use_json_mode: bool,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _mock_response(self, stage: str, prompt_text: str, temperature: float | None = None) -> Dict[str, Any]:
        temp_str = "None" if temperature is None else f"{float(temperature):.3f}"
        h = int(stable_hash(stage + "||" + prompt_text + "||" + temp_str)[:8], 16)
        keys = stage_keys(stage)
        result: Dict[str, Any] = {}
        for i, key in enumerate(keys):
            score = (h >> (i % 8)) % 5
            conf_raw = 0.55 + (((h >> ((i + 3) % 8)) % 40) / 100.0)
            result[key] = int(score)
            result[f"{key}_confidence"] = round(min(conf_raw, 0.98), 3)
            result[f"{key}_rationale"] = f"mock rationale {i+1}"
        result["summary"] = f"mock_{stage}_{h % 10000}"
        return sanitize_stage_result(stage, result)

    def chat_json(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> Dict[str, Any]:

        if self.mock_mode:
            return self._mock_response(stage, system_prompt + user_prompt, temperature=temperature)


        if not self.api_key or "YOUR_" in self.api_key or "REPLACE" in self.api_key:
            raise ValueError("Set a valid API key in API_Config.json or enable mock_mode.")


        last_error: Exception | None = None

        use_json_mode = self.use_json_mode


        for attempt in range(1, self.max_retries + 1):
            try:

                payload = self._build_payload(
                    system_prompt,
                    user_prompt,
                    use_json_mode=use_json_mode,
                    temperature=temperature,
                )


                resp = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )


                if resp.status_code >= 400:
                    text = resp.text


                    if use_json_mode and "response_format" in text:
                        use_json_mode = False
                        time.sleep(self.sleep_seconds)
                        continue


                    raise RuntimeError(f"DeepSeek request failed with status={resp.status_code}; body={text[:500]}")


                data = resp.json()

                content = data["choices"][0]["message"]["content"]

                parsed = extract_first_json(content)


                time.sleep(self.sleep_seconds)


                return sanitize_stage_result(stage, parsed)

            except Exception as exc:

                last_error = exc

                wait = min(2.0 * attempt, 8.0) + random.random() * 0.3
                time.sleep(wait)


        raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")

    def build_sampling_plan(self, stage_name: str | None = None) -> List[float]:

        num_samples = self.semantic_num_samples

        temps = self.semantic_temperature_list


        if stage_name:

            stage_num = self.stage_num_samples.get(stage_name)
            if stage_num is not None:
                try:

                    num_samples = int(stage_num)
                except Exception:

                    num_samples = self.semantic_num_samples


            stage_temps = self.stage_temperature_list.get(stage_name)
            if isinstance(stage_temps, list) and stage_temps:
                try:

                    temps = [float(x) for x in stage_temps]
                except Exception:

                    temps = self.semantic_temperature_list


        num_samples = max(1, int(num_samples))

        plan: List[float] = []


        for i in range(num_samples):

            plan.append(float(temps[i % len(temps)]))


        return plan
