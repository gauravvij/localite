"""Needle 26M tool name validation client.

Wraps the Needle 26M function-calling model for pre-dispatch
tool name validation in the AgentLoop. Gracefully degrades
if the Needle checkpoint or dependencies are unavailable.
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NeedleClient:
    """Lightweight wrapper around Needle 26M for tool name validation.

    Loads the Needle model once on init and provides validate_tool_call()
    that returns a corrected tool name given the query + tool schemas.
    Gracefully degrades (returns original call) if Needle is not available.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, disabled: bool = False):
        self._model = None
        self._params = None
        self._tokenizer = None
        self._config = None
        self._disabled = disabled

        # Find checkpoint if not specified
        if checkpoint_path is None:
            candidates = [
                os.path.join(
                    os.path.dirname(__file__),
                    "../../needle/checkpoints/needle.pkl",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../needle/checkpoints/checkpoint_epoch3.pkl",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "../../needle/checkpoints/checkpoint.pkl",
                ),
            ]
            for c in candidates:
                resolved = os.path.realpath(os.path.abspath(c))
                if os.path.exists(resolved):
                    checkpoint_path = resolved
                    break

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                import jax

                # Do NOT force CPU — let JAX auto-detect GPU/CUDA.
                # Forcing "cpu" here caused >55 min XLA compilation on GPU nodes
                # because JAX compiled the entire decode graph for CPU XLA backend.

                from needle import (
                    load_checkpoint,
                    SimpleAttentionNetwork,
                    get_tokenizer,
                )

                self._params, self._config = load_checkpoint(checkpoint_path)
                self._model = SimpleAttentionNetwork(self._config)
                self._tokenizer = get_tokenizer()

                param_count = sum(
                    x.size for x in jax.tree.leaves(self._params)
                )
                logger.info(
                    "Needle model loaded: %s checkpoint, %d params",
                    os.path.basename(checkpoint_path),
                    param_count,
                )
                self._warm_up()
            except Exception as e:
                logger.warning(
                    "Needle init failed: %s. Graceful degradation.", e
                )
                self._disabled = True
        else:
            logger.warning(
                "Needle checkpoint not found at %s. Graceful degradation.",
                checkpoint_path,
            )
            self._disabled = True

    @property
    def disabled(self) -> bool:
        return self._disabled

    def _warm_up(self):
        """Run a dummy generation to JIT-compile the model."""
        from needle import generate

        try:
            _ = generate(
                self._model,
                self._params,
                self._tokenizer,
                "test",
                tools='[{"name":"test","description":"test","parameters":{}}]',
                max_gen_len=128,
                max_enc_len=1024,
                stream=False,
            )
            logger.info("Needle warm-up complete")
        except Exception as e:
            logger.debug("Needle warm-up skipped: %s", e)

    def _build_tools_json(self, tools_dict: dict) -> str:
        """Convert tool schemas to Needle's compact tools JSON format.

        Needle expects short, clean tool definitions (like official README):
          [{"name": "...", "description": "...", "parameters": {...}}]

        Descriptions are trimmed to 1 sentence (max 120 chars) to stay within
        Needle's default 1024-token encoder limit. Parameter descriptions
        are trimmed to 80 chars. No extra fields beyond type/description.
        """
        tools = []
        for name, tool in tools_dict.items():
            desc = getattr(tool, "description", "") or ""
            short_desc = desc.split(".")[0] + "." if desc else ""
            if len(short_desc) > 120:
                short_desc = short_desc[:117] + "..."

            entry: dict[str, Any] = {
                "name": name,
                "description": short_desc,
            }

            schema = getattr(tool, "parameters", None) or {}
            if isinstance(schema, dict) and "properties" in schema:
                raw_params = schema["properties"]
            else:
                raw_params = schema

            # Needle's ToolConstraints iterates over parameters keys directly
            # to build param tries, so params must be flat (NOT type:object/properties):
            #   {"path": {"type": "string", "description": "..."}}
            compact_params = {}
            for p_name, p_info in raw_params.items():
                if isinstance(p_info, dict):
                    compact: dict[str, Any] = {
                        "type": p_info.get("type", "string"),
                    }
                    p_desc = p_info.get("description", "")
                    if p_desc:
                        compact["description"] = p_desc[:80]
                    if p_info.get("required"):
                        compact["required"] = True
                    compact_params[p_name] = compact

            entry["parameters"] = compact_params
            tools.append(entry)

        return json.dumps(tools)

    def validate_tool_call(
        self,
        query: str,
        tools_dict: dict,
        proposed_tool_name: str,
        proposed_args: dict,
    ) -> dict:
        """Validate/override the tool name using Needle.

        Passes the raw objective to Needle as-is (matching the official
        training format where query=natural language). Needle returns the
        correct tool call. If Needle's tool name differs from Gemma's,
        the correction is logged and applied.

        Args:
            query: The user's objective (full task description).
            tools_dict: dict of tool_name -> tool_instance
                with .description, .parameters attributes.
            proposed_tool_name: The name Gemma proposed to call.
            proposed_args: The arguments Gemma proposed.

        Returns:
            dict with validated "name" and "arguments" keys.
            If Needle is disabled or fails, returns the original call.
        """
        if self._disabled or self._model is None:
            return {"name": proposed_tool_name, "arguments": proposed_args}

        # Build compact tools JSON (short descriptions to fit 1024-token limit)
        tools_json = self._build_tools_json(tools_dict)

        try:
            from needle import generate

            # Use raw query (Needle was trained on natural language queries)
            # Increase encoder limit to 2048 to comfortably fit all tool schemas

            # Debug: log actual token counts to verify encoder input fits
            try:
                q_toks = self._tokenizer.encode(query)
                t_toks = self._tokenizer.encode(tools_json)
                logger.info(
                    "Needle input: query=%d tokens, tools=%d tokens, total=%d, max_enc_len=%d",
                    len(q_toks), len(t_toks), len(q_toks) + 1 + len(t_toks), 2048,
                )
            except Exception:
                pass

            result_text = generate(
                self._model,
                self._params,
                self._tokenizer,
                query,
                tools=tools_json,
                max_gen_len=128,
                max_enc_len=1024,
                stream=False,
            )

            if result_text and result_text.strip():
                # Strip any <tool_call> prefix
                if result_text.startswith("<tool_call>"):
                    result_text = result_text[len("<tool_call>") :]

                needle_output = json.loads(result_text)
                # Needle returns a list of tool calls; take the first one
                if isinstance(needle_output, list) and len(needle_output) > 0:
                    needle_output = needle_output[0]
                if isinstance(needle_output, dict) and "name" in needle_output:
                    validated_name = needle_output["name"]
                    validated_args = needle_output.get(
                        "arguments",
                        needle_output.get("args", proposed_args),
                    )

                    if validated_name != proposed_tool_name:
                        logger.info(
                            "Needle corrected tool name: '%s' -> '%s'",
                            proposed_tool_name,
                            validated_name,
                        )

                    return {
                        "name": validated_name,
                        "arguments": validated_args,
                    }

        except Exception as e:
            logger.warning(
                "Needle validation failed: %s. Using original.", e
            )

        return {"name": proposed_tool_name, "arguments": proposed_args}