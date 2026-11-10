"""本地模型推理引擎（transformers + 可选 LoRA 适配器）与一个无需显存的 mock 引擎。"""

from __future__ import annotations

import os
import time


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        device: str | None = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 768,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.name = "ft" if adapter_path else "base"
        self._model = None
        self._tok = None
        self.n_calls = 0
        self.gen_seconds = 0.0

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        kwargs = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        if self.device == "cuda":
            kwargs["device_map"] = {"": 0}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()
        print(f"[engine] loaded {self.model_path} adapter={self.adapter_path} in {time.time() - t0:.1f}s", flush=True)

    def unload(self) -> None:
        import gc

        self._model = None
        self._tok = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        import torch

        self.load()
        with torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tok([text], return_tensors="pt").to(self._model.device)
            t0 = time.time()
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature if temperature is None else max(temperature, 1e-5),
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        self.n_calls += 1
        self.gen_seconds += time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._tok.decode(gen, skip_special_tokens=True).strip()


class MockLLM:
    """不依赖模型的确定性引擎，用于前端联调与快速冒烟测试。"""

    name = "mock"

    def __init__(self, *_, **__):
        self.n_calls = 0
        self.gen_seconds = 0.0

    def load(self) -> None:
        return

    def chat(self, messages: list[dict], max_new_tokens: int | None = None, **kw) -> str:
        import json as _json
        import re

        user = messages[-1]["content"]
        ids = re.findall(r'"id":\s*"([^"]+)"', user)
        is_leaf = "最后一层" in user
        rows = []
        for i, nid in enumerate(ids):
            rows.append(
                {
                    "id": nid,
                    "line": f"（mock）第{i + 1}句台词：这件事不该在这儿说。",
                    "emotion": "戒备",
                    "options": [] if is_leaf else ["你在隐瞒什么？", "我可以帮你。"],
                }
            )
        self.n_calls += 1
        return _json.dumps({"nodes": rows}, ensure_ascii=False)


def get_engine(kind: str = "base", model_path: str | None = None, adapter_path: str | None = None):
    if kind == "mock":
        return MockLLM()
    base = model_path or os.environ.get(
        "NPC_MODEL_PATH", os.path.join("models", "Qwen2.5-1.5B-Instruct")
    )
    adapter = adapter_path
    if kind == "ft" and not adapter:
        adapter = os.environ.get("NPC_ADAPTER_PATH", os.path.join("training", "output", "lora"))
    return LocalLLM(base, adapter if kind == "ft" else None)
