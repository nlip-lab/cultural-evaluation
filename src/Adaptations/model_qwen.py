"""Qwen2.5-72B-Instruct-AWQ adapter (Alibaba). AWQ 4-bit, TP=2, ~38-42 GB VRAM."""

import gc
import torch
from vllm import LLM, SamplingParams

class QwenAdapter:
    MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    BATCH_SIZE = 4

    def __init__(self):
        self._llm = None

    @property
    def model_id(self) -> str:
        """Return serving model identifier."""
        return self.MODEL_NAME

    def load(self) -> None:
        """Load model weights into vLLM runtime."""
        print(f"  [qwen] Loading {self.MODEL_NAME} (AWQ 4-bit, TP=2, vLLM)", flush=True)
        self._llm = LLM(
            model=self.MODEL_NAME,
            tensor_parallel_size=2,
            dtype="auto",
            quantization="awq_marlin",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            enforce_eager=True,
        )
        _log_vram_all("qwen")

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate one response for one chat conversation."""
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            **_sampling_kwargs(temperature),
        )
        outputs = self._llm.chat(messages, sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Generate for a batch of conversations. Returns list of outputs in same order."""
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            **_sampling_kwargs(temperature),
        )
        outputs = self._llm.chat(batch_messages, sampling_params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]

    def unload(self) -> None:
        """Release model resources and clear VRAM cache."""
        _cleanup_vllm(self)
        print("  [qwen] Unloaded.", flush=True)

def _sampling_kwargs(temperature: float) -> dict:
    """Build deterministic or sampling kwargs from temperature."""
    if temperature > 0.0:
        return {"temperature": temperature, "top_p": 0.9}
    return {"temperature": 0, "top_p": 1.0}

def _log_vram_all(tag: str) -> None:
    """Log allocated VRAM for all visible GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gb = torch.cuda.memory_allocated(i) / 1024**3
            if gb > 0.1:
                print(f"  [{tag}] GPU {i} VRAM allocated: {gb:.1f} GB", flush=True)

def _cleanup_vllm(adapter) -> None:
    """Dispose adapter vLLM object and clear distributed state."""
    if adapter._llm is not None:
        del adapter._llm
        adapter._llm = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
