"""AsyncOllamaClient — Async HTTPX-based client for Ollama's /api/chat endpoint.

Ports patterns from src/eval_harness.py but uses httpx for async support
and proper streaming via SSE.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx


logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 30.0
DEFAULT_BASE_URL = "http://localhost:11434"


def strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output.

    LFM2.5 uses plain-text headers: 'thinking' then JSON (or text) after 'response'.
    LFM2.5 also uses <thinking>...</thinking> <response>...</response> XML format.
    Gemma 4 uses <|channel>thought</channel|> tags.
    Also handles standalone '<tool_call_start|>' blocked format.
    """
    if not text:
        return ""

    # Remove Gemma 4 style: <|channel>thought</channel|> blocks
    text = re.sub(r'<\|channel>thought<\|?[^>]*>.*?</channel\|?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|channel>thought.*?</channel\|?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|channel\|?>.*?</channel\|?>', '', text, flags=re.DOTALL)

    # Remove <thinking>...</thinking> blocks (XML style)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

    # Handle orphaned </thinking> tags
    text = text.replace('</thinking>', '')
    text = text.replace('<thinking>', '')

    # If <response> tag exists, take everything after it (XML style)
    if '<response>' in text:
        text = text.split('<response>', 1)[1]

    # Remove closing </response> tag if present
    text = text.replace('</response>', '')

    # Handle LFM2.5 plain-text "thinking\n...\n response\n" blocks
    # Simple approach: find "thinking" at a line boundary, remove everything
    # from there through the next "response" line boundary
    text = re.sub(
        r'(?:^|\n)\s*thinking\b.*?(?:\n\s*response\s*\n)',
        '\n',
        text,
        flags=re.DOTALL,
    )

    # Strip standalone " response" headers leftover after removal
    text = re.sub(r'\n\s*response\s*\n', '\n', text)
    text = re.sub(r'^response\s*\n?', '', text)

    # Preserve tool call markers — the parser needs them
    # Only strip dangling <s>, </s>, <|im_start|>, <|im_end|> tags
    text = re.sub(r'<\|?(im_start|im_end|/s)\|?>', '', text)

    return text.strip()


class AsyncOllamaClient:
    """Async client for Ollama's /api/chat endpoint using httpx."""

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        has_thinking_tags: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.has_thinking_tags = has_thinking_tags

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncGenerator[str, None]:
        """Send a chat request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            stream: If True, returns an async generator yielding token strings.
            options: Optional Ollama options dict (e.g. temperature).

        Returns:
            If stream=False: The stripped response text.
            If stream=True: An async generator yielding token strings.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
            httpx.HTTPStatusError: If the API returns an error status.
            httpx.TimeoutException: If the request times out.
        """
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        # DEBUG: log payload structure before request (not full messages)
        logger.debug(
            "CHAT REQUEST — model=%s, num_messages=%d, stream=%s, options=%s",
            self.model_name,
            len(messages),
            stream,
            options,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                if stream:
                    return self._stream_response(response)
                else:
                    data = response.json()
                    msg = data.get("message", {})
                    raw = msg.get("content", "")
                    # DEBUG: log raw msg keys and sizes
                    content_len = len(raw)
                    thinking_raw = msg.get("thinking", "")
                    thinking_len = len(thinking_raw)
                    done_reason = data.get("done_reason", "")
                    eval_count = data.get("eval_count", 0)
                    msg_keys = list(msg.keys())
                    logger.debug(
                        "CHAT RESPONSE RAW — content_len=%d, thinking_len=%d, "
                        "done_reason=%s, eval_count=%d, msg_keys=%s",
                        content_len,
                        thinking_len,
                        done_reason,
                        eval_count,
                        msg_keys,
                    )
                    # Gemma 4 E4B puts tool-call JSON in the native Ollama
                    # `thinking` field when conversation history is complex.
                    # Fall through to `thinking` if `content` is empty.
                    path_taken = "content"
                    if not raw.strip():
                        raw = msg.get("thinking", "")
                        path_taken = "thinking_fallback"
                    logger.debug(
                        "CHAT PATH — path=%s, has_thinking_tags=%s",
                        path_taken,
                        self.has_thinking_tags,
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
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is Ollama running? Error: {e}"
                ) from e
            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Ollama request timed out after {self.timeout}s: {e}"
                ) from e

    async def _stream_response(
        self, response: httpx.Response
    ) -> AsyncGenerator[str, None]:
        """Parse SSE stream from Ollama and yield tokens."""
        full_text = ""
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_text += content
                    # Yield raw tokens as they come
                    yield content
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue

        # After streaming completes, if model has thinking tags, we need to
        # strip thinking from the full accumulated text. But since we already
        # yielded raw tokens, the caller should use strip_thinking on final.
        # We yield a special sentinel with the stripped version.
        if self.has_thinking_tags and full_text.strip():
            stripped = strip_thinking(full_text)
            yield ("__STRIPPED__", stripped)