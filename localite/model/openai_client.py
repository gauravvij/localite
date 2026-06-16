"""AsyncOpenAIClient — Async HTTPX-based client for OpenAI-compatible /v1/chat/completions endpoints.

Designed for use with vLLM, TGI, or any server exposing an OpenAI-compatible API.
Same interface as AsyncOllamaClient for drop-in replacement in swe_runner.py.
"""

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from localite.model.client import strip_thinking


logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 300.0


class AsyncOpenAIClient:
    """Async client for OpenAI-compatible /v1/chat/completions using httpx."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        has_thinking_tags: bool = False,
        disable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.has_thinking_tags = has_thinking_tags
        self.disable_thinking = disable_thinking

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncGenerator[str, None]:
        """Send a chat request to an OpenAI-compatible endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            stream: If True, returns an async generator yielding token strings.
            options: Optional dict. Maps 'num_predict' to 'max_tokens' (OpenAI param).

        Returns:
            If stream=False: The stripped response text.
            If stream=True: An async generator yielding token strings.

        Raises:
            ConnectionError: If the endpoint is unreachable.
            httpx.HTTPStatusError: If the API returns an error status.
            TimeoutError: If the request times out.
        """
        # Build the payload per OpenAI spec
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }

        # Suppress chain-of-thought thinking for Qwen3 models
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Map options to OpenAI-compatible parameters
        if options:
            openai_params = {}
            # Map num_predict -> max_tokens (most common mapping)
            if "num_predict" in options:
                openai_params["max_tokens"] = options["num_predict"]
            # Also pass temperature, top_p, etc. directly if present
            for k in ("temperature", "top_p", "stop", "frequency_penalty", "presence_penalty"):
                if k in options:
                    openai_params[k] = options[k]
            if openai_params:
                payload.update(openai_params)

        # Safety check: if max_tokens is set, ensure prompt + max_tokens fits within a reasonable
        # context window. Estimate tokens roughly at 1 char = ~0.5 tokens for mixed content.
        # vLLM will reject if prompt_tokens + max_tokens > max_model_len.
        if "max_tokens" in payload:
            raw_text = "".join(m.get("content", "") for m in messages)
            estimated_prompt_tokens = len(raw_text) // 4  # conservative: 4 chars per token for safety margin
            effective_max_tokens = payload["max_tokens"]
            if estimated_prompt_tokens + effective_max_tokens > 16000:  # max_model_len is ~16384
                reduced = max(512, 16000 - estimated_prompt_tokens - 256)  # leave 256 token margin
                if reduced < effective_max_tokens:
                    logger.warning(
                        "Estimated total tokens (%d prompt + %d max = %d) exceeds 16K limit. "
                        "Reducing max_tokens from %d to %d.",
                        estimated_prompt_tokens, effective_max_tokens,
                        estimated_prompt_tokens + effective_max_tokens,
                        effective_max_tokens, reduced,
                    )
                    payload["max_tokens"] = reduced

        logger.debug(
            "CHAT REQUEST — model=%s, num_messages=%d, stream=%s, options=%s, max_tokens=%s",
            self.model_name,
            len(messages),
            stream,
            options,
            payload.get("max_tokens"),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                if stream:
                    return self._stream_response(response)
                else:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raw = ""
                        logger.warning("CHAT RESPONSE — no choices in response: %s", data)
                    else:
                        raw = choices[0].get("message", {}).get("content", "")

                    logger.debug(
                        "CHAT RESPONSE RAW — content_len=%d",
                        len(raw),
                    )

                    if self.has_thinking_tags:
                        before = len(raw)
                        stripped = strip_thinking(raw)
                        after = len(stripped)
                        if before != after:
                            logger.debug(
                                "CHAT STRIP_THINKING — before=%d, after=%d, diff=%d",
                                before,
                                after,
                                before - after,
                            )
                        return stripped
                    return raw

            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Cannot connect to endpoint at {self.base_url}. "
                    f"Error: {e}"
                ) from e
            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Request timed out after {self.timeout}s: {e}"
                ) from e

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[str, None]:
        """Parse SSE stream from OpenAI-compatible endpoint and yield tokens."""
        full_text = ""
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            # Handle SSE format: "data: {...}"
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        yield content
                except json.JSONDecodeError:
                    continue

        # After streaming completes, strip thinking from accumulated text
        if self.has_thinking_tags and full_text.strip():
            stripped = strip_thinking(full_text)
            yield ("__STRIPPED__", stripped)