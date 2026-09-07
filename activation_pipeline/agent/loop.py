"""Multi-turn agent loop: plan → <tool_call> → real tool result → continue (Cursor-like)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from activation_pipeline.loader import LoadedModel

from .tools import ToolRegistry, hermes_tools_block, tools_for_prompt

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


@dataclass
class AgentRunResult:
    transcript_id: str
    domain: str
    task: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]
    n_tool_calls: int
    stop_reason: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_transcript(self) -> dict[str, Any]:
        return {
            "transcript_id": self.transcript_id,
            "domain": self.domain,
            "tool_format": "qwen3_hermes",
            "source": "agentic_loop",
            "tools": self.tools,
            "messages": self.messages,
            "meta": {
                "task": self.task,
                "n_tool_calls": self.n_tool_calls,
                "stop_reason": self.stop_reason,
                **self.meta,
            },
        }


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            # Flat form: {"name": "read_file", "path": "x"} — common small-model slip
            args = {k: v for k, v in obj.items() if k not in {"name", "tool", "function"}}
        if not isinstance(args, dict):
            continue
        if isinstance(name, str) and name:
            calls.append({"name": name, "arguments": args, "raw": m.group(0)})
    return calls


def _build_messages(
    *,
    system: str,
    user_task: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_task},
    ]


def _chat_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply chat template; disable Qwen3 thinking when supported."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def generate_assistant(
    loaded: LoadedModel,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    assistant_prefill: str | None = None,
    return_stats: bool = False,
) -> str | tuple[str, dict[str, Any]]:
    """Generate the next assistant turn.

    If ``assistant_prefill`` is set, teacher-force those tokens after the
    generation prompt and continue sampling; the returned string is
    ``prefill + continuation``.
    If ``return_stats``, also return mean token logprob / PPL of new tokens.
    """
    tok = loaded.tokenizer
    prompt = _chat_text(tok, messages)
    if assistant_prefill:
        prompt = prompt + assistant_prefill
    encoded = tok(prompt, return_tensors="pt")
    first = next(loaded.model.parameters())
    encoded = {k: v.to(first.device) for k, v in encoded.items()}
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False
    if return_stats:
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True
    out = loaded.model.generate(**encoded, **gen_kwargs)
    if return_stats:
        seq = out.sequences[0]
        new_tokens = seq[encoded["input_ids"].shape[1] :]
        scores = out.scores or ()
        logps: list[float] = []
        for logit, tid in zip(scores, new_tokens.tolist()):
            lp = torch.log_softmax(logit[0].float(), dim=-1)[int(tid)]
            logps.append(float(lp))
        mean_lp = float(sum(logps) / len(logps)) if logps else float("nan")
        stats = {
            "n_new_tokens": int(new_tokens.numel()),
            "mean_logprob": mean_lp,
            "ppl": float(math.exp(-mean_lp)) if logps else float("nan"),
        }
    else:
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        stats = None
    text = tok.decode(new_tokens, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if assistant_prefill:
        text = assistant_prefill + text
    if return_stats:
        return text, stats or {}
    return text


def run_agent_task(
    loaded: LoadedModel,
    *,
    transcript_id: str,
    domain: str,
    task: str,
    workspace: Path,
    max_turns: int = 6,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> AgentRunResult:
    """Run a Cursor-like agent: generate → execute tools → append results → repeat."""
    tools = ToolRegistry(workspace)
    system = (
        "You are a coding agent operating in a real repository workspace "
        "(similar to Cursor Agent). Use tools to inspect and change files; "
        "do not invent file contents. Paths are relative to the workspace root "
        "(never absolute paths like /usr/...).\n\n"
        + hermes_tools_block()
        + "\n\nExample (format only):\n"
        "I'll search for the symbol.\n"
        "<tool_call>\n"
        '{"name": "search_code", "arguments": {"query": "RATE_LIMIT"}}\n'
        "</tool_call>"
    )
    messages = _build_messages(system=system, user_task=task)
    n_calls = 0
    stop_reason = "max_turns"

    for _ in range(max_turns):
        assistant = generate_assistant(
            loaded,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        messages.append({"role": "assistant", "content": assistant})
        calls = parse_tool_calls(assistant)
        if not calls:
            stop_reason = "final_answer"
            break

        # Execute all tool calls in the turn; return concatenated responses
        # (one user message per assistant turn — matches transcript schema).
        chunks: list[str] = []
        for call in calls:
            result = tools.execute(call["name"], call["arguments"])
            n_calls += 1
            chunks.append(f"<tool_response>\n{result}\n</tool_response>")
        messages.append({"role": "user", "content": "\n".join(chunks)})
    else:
        stop_reason = "max_turns"

    return AgentRunResult(
        transcript_id=transcript_id,
        domain=domain,
        task=task,
        messages=messages,
        tools=tools_for_prompt(),
        n_tool_calls=n_calls,
        stop_reason=stop_reason,
        meta={
            "model_key": loaded.spec.key,
            "hf_id": loaded.spec.hf_id,
            "workspace": str(workspace.resolve()),
            "max_turns": max_turns,
        },
    )
