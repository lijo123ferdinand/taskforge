"""Per-run token and cost budget with hard stops.

Two enforcement points:

1. **Pre-call**: refuse a call whose *estimated* prompt alone would cross the limit.
   This catches the expensive case (a huge context) before any money is spent.
2. **Post-call**: charge the provider-reported usage; if the run is now over, raise.

The overshoot is therefore bounded by one call per concurrently running worker.
Getting to zero overshoot would mean reserving ``max_tokens`` per call up front,
which starves parallel workers long before the budget is actually used, so we
take the bounded overshoot and document it.

Cost is only tracked when you pass prices. We deliberately ship no price table:
prices change, and a stale hard-coded price is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .providers.base import Usage

__all__ = ["Budget", "BudgetExceeded"]


class BudgetExceeded(RuntimeError):
    """The run hit a hard limit. The engine stops scheduling and checkpoints."""


@dataclass
class Budget:
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_cost_usd is not None and not (self.usd_per_mtok_in or self.usd_per_mtok_out):
            raise ValueError("max_cost_usd needs usd_per_mtok_in/usd_per_mtok_out prices")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * self.usd_per_mtok_in + self.output_tokens * self.usd_per_mtok_out) / 1e6

    def check_before_call(self, estimated_input_tokens: int) -> None:
        if self.max_tokens is not None and self.total_tokens + estimated_input_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget: {self.total_tokens} used + ~{estimated_input_tokens} for next call > {self.max_tokens}")
        if self.max_cost_usd is not None:
            projected = self.cost_usd + estimated_input_tokens * self.usd_per_mtok_in / 1e6
            if projected > self.max_cost_usd:
                raise BudgetExceeded(f"cost budget: ${projected:.4f} projected > ${self.max_cost_usd:.4f}")

    def charge(self, usage: Usage, agent: str = "worker") -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.calls += 1
        self.by_agent[agent] = self.by_agent.get(agent, 0) + usage.total
        if self.max_tokens is not None and self.total_tokens > self.max_tokens:
            raise BudgetExceeded(f"token budget exhausted: {self.total_tokens} > {self.max_tokens}")
        if self.max_cost_usd is not None and self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"cost budget exhausted: ${self.cost_usd:.4f} > ${self.max_cost_usd:.4f}")

    def to_dict(self) -> dict[str, Any]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "calls": self.calls,
                "by_agent": dict(self.by_agent), "cost_usd": round(self.cost_usd, 6)}

    def restore(self, data: dict[str, Any]) -> None:
        """Resume keeps counting from where the crashed run stopped: a crash must not reset spend."""
        self.input_tokens = int(data.get("input_tokens", 0))
        self.output_tokens = int(data.get("output_tokens", 0))
        self.calls = int(data.get("calls", 0))
        self.by_agent = dict(data.get("by_agent", {}))
