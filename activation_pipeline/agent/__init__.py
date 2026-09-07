"""Cursor-like agentic loop for collecting real tool-use transcripts."""

from .loop import AgentRunResult, run_agent_task
from .tasks import CODING_TASKS, AgentTask
from .tools import TOOL_SPECS, ToolRegistry, hermes_tools_block, tools_for_prompt

__all__ = [
    "AgentRunResult",
    "AgentTask",
    "CODING_TASKS",
    "TOOL_SPECS",
    "ToolRegistry",
    "hermes_tools_block",
    "run_agent_task",
    "tools_for_prompt",
]
