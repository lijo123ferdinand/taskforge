"""Human-in-the-loop approval gates for chosen tools.

An approver is any async callable ``(ApprovalRequest) -> bool``. A denial is not
an exception: the model receives "denied by approver" as the tool result and can
adapt (explain, pick another route), exactly as it would with a human reviewer.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

__all__ = ["ApprovalRequest", "Approver", "auto_approve", "deny_all", "ConsoleApprover", "PolicyApprover"]


@dataclass(frozen=True)
class ApprovalRequest:
    task_id: str
    tool: str
    arguments: dict[str, Any]


Approver = Callable[[ApprovalRequest], Awaitable[bool]]


async def auto_approve(request: ApprovalRequest) -> bool:
    return True


async def deny_all(request: ApprovalRequest) -> bool:
    return False


class PolicyApprover:
    """Approve with a plain predicate, e.g. ``lambda r: r.arguments["path"].startswith("output/")``."""

    def __init__(self, predicate: Callable[[ApprovalRequest], bool]) -> None:
        self.predicate = predicate

    async def __call__(self, request: ApprovalRequest) -> bool:
        return bool(self.predicate(request))


class ConsoleApprover:
    """Ask on the terminal. A lock serialises prompts: with parallel workers, two
    simultaneous questions would otherwise interleave and the answer would be ambiguous."""

    def __init__(self, stream: Any = None) -> None:
        self._lock = asyncio.Lock()
        self.stream = stream or sys.stderr

    async def __call__(self, request: ApprovalRequest) -> bool:
        async with self._lock:
            preview = json.dumps(request.arguments)[:400]
            print(f"\n[approval] task {request.task_id} wants {request.tool}({preview})", file=self.stream)
            answer = await asyncio.to_thread(input, "approve? [y/N] ")
            return answer.strip().lower() in ("y", "yes")
