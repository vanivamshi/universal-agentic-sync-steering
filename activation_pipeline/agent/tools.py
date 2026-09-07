"""Cursor-like tools for a sandboxed workspace (read / write / search / shell / list)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file under the workspace.",
        parameters={"path": {"type": "string", "description": "Relative path"}},
    ),
    ToolSpec(
        name="write_file",
        description="Write UTF-8 contents to a file under the workspace (creates parents).",
        parameters={
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    ),
    ToolSpec(
        name="list_dir",
        description="List files and directories at a relative path (non-recursive).",
        parameters={"path": {"type": "string", "description": "Relative dir; use '.' for root"}},
    ),
    ToolSpec(
        name="search_code",
        description="Search workspace text files for a regex/substring; return matching lines.",
        parameters={
            "query": {"type": "string"},
            "path": {"type": "string", "description": "Optional subdirectory", "optional": True},
        },
    ),
    ToolSpec(
        name="run_command",
        description="Run a short shell command with cwd=workspace. Prefer for tests/imports.",
        parameters={"cmd": {"type": "string"}},
    ),
]


def tools_for_prompt() -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in TOOL_SPECS
    ]


def hermes_tools_block() -> str:
    """Human-readable tool card for models without native tool templates."""
    lines = [
        "You have these tools. To call one, emit exactly:",
        "<tool_call>",
        '{"name": "<tool-name>", "arguments": { ... }}',
        "</tool_call>",
        "You may include brief planning prose before a tool call.",
        "After tool results arrive, continue until the user task is done.",
        "Available tools:",
    ]
    for t in TOOL_SPECS:
        lines.append(f"- {t.name}: {t.description} params={json.dumps(t.parameters)}")
    return "\n".join(lines)


class ToolRegistry:
    """Execute tools against a workspace root (path sandbox)."""

    def __init__(self, workspace: Path, *, command_timeout_s: float = 15.0) -> None:
        self.workspace = workspace.resolve()
        self.command_timeout_s = command_timeout_s
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "search_code": self._search_code,
            "run_command": self._run_command,
        }

    def _resolve(self, rel: str) -> Path:
        raw = (self.workspace / rel).resolve()
        if not str(raw).startswith(str(self.workspace)):
            raise PermissionError(f"path escapes workspace: {rel}")
        return raw

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._handlers:
            return f"ERROR: unknown tool {name!r}. Known: {sorted(self._handlers)}"
        try:
            return self._handlers[name](arguments or {})
        except Exception as e:  # noqa: BLE001 — surface tool errors to the agent
            return f"ERROR: {type(e).__name__}: {e}"

    def _read_file(self, args: dict[str, Any]) -> str:
        path = self._resolve(str(args["path"]))
        if not path.is_file():
            return f"ERROR: not a file: {args['path']}"
        text = path.read_text(encoding="utf-8")
        if len(text) > 20_000:
            return text[:20_000] + "\n...[truncated]"
        return text

    def _write_file(self, args: dict[str, Any]) -> str:
        path = self._resolve(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        path.write_text(content, encoding="utf-8")
        return f"wrote {args['path']} ({len(content)} bytes)"

    def _list_dir(self, args: dict[str, Any]) -> str:
        path = self._resolve(str(args.get("path", ".")))
        if not path.is_dir():
            return f"ERROR: not a directory: {args.get('path', '.')}"
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(names) if names else "(empty)"

    def _search_code(self, args: dict[str, Any]) -> str:
        query = str(args["query"])
        root = self._resolve(str(args.get("path") or "."))
        if not root.exists():
            return f"ERROR: path not found: {args.get('path')}"
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))
        hits: list[str] = []
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if not f.is_file():
                continue
            if f.suffix.lower() not in {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml"}:
                continue
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            rel = f.relative_to(self.workspace)
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= 40:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "(no matches)"

    def _run_command(self, args: dict[str, Any]) -> str:
        cmd = str(args["cmd"])
        if any(x in cmd for x in (";", "&&", "||", "`", "$(", ">", "<", "|")):
            return "ERROR: shell metacharacters blocked for safety"
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out"
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.strip() or f"(exit {proc.returncode}, empty output)"
        if len(out) > 12_000:
            out = out[:12_000] + "\n...[truncated]"
        return out
