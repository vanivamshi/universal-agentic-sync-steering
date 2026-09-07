"""Abstract base for benchmark domains."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ForbiddenAction:
    """A specific tool-call behavior that should not happen."""

    id: str
    description: str
    category: str  # "contract" or "intent"
    check: object  # (tool_name: str, tool_args: dict) -> bool

    def matches(self, tool_name: str, tool_args: dict) -> bool:
        return self.check(tool_name, tool_args)


class BenchmarkDomain(ABC):
    """Interface that each domain must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain identifier (e.g., 'pharma', 'finance')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def role(self) -> str:
        """The agent's role in this domain."""

    @property
    @abstractmethod
    def regulation(self) -> str:
        """Key regulations (e.g., 'HIPAA, ICH-GCP')."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the LLM."""

    @property
    @abstractmethod
    def tools_for_llm(self) -> list[dict]:
        """OpenAI-format tool definitions for the LLM."""

    @property
    @abstractmethod
    def tool_functions(self) -> dict[str, callable]:
        """Map of tool_name -> callable function."""

    @property
    @abstractmethod
    def scenarios(self) -> dict[str, dict]:
        """Scenario definitions with forbidden_actions."""

    @property
    @abstractmethod
    def pii_markers(self) -> list[str]:
        """PII marker strings for text safety scoring."""

    @property
    @abstractmethod
    def contracts_path(self) -> Path:
        """Path to Edictum contracts YAML."""

    @property
    @abstractmethod
    def principal_role(self) -> str:
        """Default principal role for this domain."""

    def make_principal(self, role: str | None = None, ticket_ref: str | None = None):
        """Create an Edictum Principal for this domain."""
        from edictum import Principal

        r = role or self.principal_role
        return Principal(
            user_id=f"benchmark-{self.name}-{r}",
            role=r,
            ticket_ref=ticket_ref,
            claims={"domain": self.name},
        )
