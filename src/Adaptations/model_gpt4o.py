"""GPT-4o adapter via OpenAI REST API. Requires OPENAI_API_KEY. Rate-limit-aware."""

import os
import time

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("openai package required: pip install openai>=1.0") from e

class GPT4oAdapter:
    """OpenAI GPT-4o adapter."""

    MODEL_ID = "gpt-4o"
    MAX_RETRIES = 5
    RETRY_BASE_DELAY = 2.0
    BATCH_SIZE = 1

    def __init__(self):
        self._client: OpenAI | None = None

    @property
    def model_id(self) -> str:
        """Return serving model identifier."""
        return self.MODEL_ID

    def load(self) -> None:
        """Initialise the OpenAI client. Validates API key is present."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )
        self._client = OpenAI(api_key=api_key)
        print(f"  [gpt4o] Client initialised (model={self.MODEL_ID})", flush=True)

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> str:
        """Send messages to the API and return the raw text response."""
        if self._client is None:
            raise RuntimeError("Call load() before generate().")

        delay = self.RETRY_BASE_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.MODEL_ID,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if attempt < self.MAX_RETRIES - 1 and any(
                    code in err for code in ["429", "500", "502", "503", "RateLimitError"]
                ):
                    print(
                        f"  [gpt4o] Attempt {attempt+1} failed ({err[:60]}), "
                        f"retrying in {delay:.1f}s ...",
                        flush=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                else:
                    raise
        return ""

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Sequential fallback for API model — no true batching."""
        return [
            self.generate(msgs, temperature, max_new_tokens)
            for msgs in batch_messages
        ]

    def unload(self) -> None:
        """No-op for API models — nothing to release."""
        self._client = None
        print("  [gpt4o] Client released.", flush=True)
