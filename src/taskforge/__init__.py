"""taskforge: a minimal, framework-free multi-agent orchestration engine."""

from .approval import ApprovalRequest, ConsoleApprover, PolicyApprover, auto_approve, deny_all
from .blackboard import Blackboard
from .budget import Budget, BudgetExceeded
from .dag import PlanError, Task, TaskGraph
from .engine import Orchestrator, RunConfig, RunResult, TaskState
from .providers import MockProvider, make_provider
from .roles import DEFAULT_ROLES, Role
from .timeline import render_timeline
from .tools import Tool, ToolRegistry, default_registry

__version__ = "0.1.0"

__all__ = ["ApprovalRequest", "ConsoleApprover", "PolicyApprover", "auto_approve", "deny_all", "Blackboard",
           "Budget", "BudgetExceeded", "PlanError", "Task", "TaskGraph", "Orchestrator", "RunConfig", "RunResult",
           "TaskState", "MockProvider", "make_provider", "DEFAULT_ROLES", "Role", "render_timeline", "Tool",
           "ToolRegistry", "default_registry", "__version__"]
