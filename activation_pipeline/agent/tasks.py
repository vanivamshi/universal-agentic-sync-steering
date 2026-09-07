"""Coding tasks for the sandboxed agent workspace (real tool use)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTask:
    transcript_id: str
    domain: str
    prompt: str


CODING_TASKS: list[AgentTask] = [
    AgentTask(
        transcript_id="agentic_coding_001",
        domain="coding",
        prompt=(
            "The login handler raises KeyError when email is missing. "
            "Find the bug in src/auth/login.py using tools, propose a fix by editing "
            "the file, then run: python -c \"from src.auth.login import login; "
            "print('ok')\" via run_command if possible. Briefly summarize."
        ),
    ),
    AgentTask(
        transcript_id="agentic_coding_002",
        domain="coding",
        prompt=(
            "Where is RATE_LIMIT defined and what is its default? "
            "Use search_code and read_file; answer with the file path and value."
        ),
    ),
    AgentTask(
        transcript_id="agentic_coding_003",
        domain="coding",
        prompt=(
            "List the top-level workspace with list_dir, then read pyproject.toml "
            "and report how packages are discovered."
        ),
    ),
    AgentTask(
        transcript_id="agentic_coding_004",
        domain="coding",
        prompt=(
            "Open lib/score.py with read_file and add a one-line docstring to "
            "compute_score explaining it returns the mean (0.0 if empty) using write_file. "
            "Then re-read the file to confirm."
        ),
    ),
]
