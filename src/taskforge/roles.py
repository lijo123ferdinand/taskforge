"""Worker roles: a system prompt plus a tool allow-list.

The allow-list is the *ceiling*. A plan can narrow it per task (``Task.tools``) but
never widen it, so a prompt-injected planner cannot hand the writer a python
sandbox. The effective allow-list is ``role.tools ∩ task.tools``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Role", "DEFAULT_ROLES", "effective_tools"]


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str
    tools: tuple[str, ...]


_COMMON = (
    "You are one worker in a multi-agent system. You get ONE task, its acceptance criteria and the outputs "
    "of the tasks it depends on. Use tools when they help; when done, reply with the final task output only "
    "(no preamble). If you get revision feedback, fix exactly what it names."
)

DEFAULT_ROLES: dict[str, Role] = {
    r.name: r for r in [
        Role("researcher", _COMMON + " Your speciality: reading source material and summarising it faithfully, "
             "citing the file you read.", ("read_file", "list_dir", "web_fetch")),
        Role("analyst", _COMMON + " Your speciality: computing exact numbers. Never estimate what you can compute "
             "with a tool.", ("python_eval", "calculator", "read_file", "list_dir", "write_file")),
        Role("writer", _COMMON + " Your speciality: turning upstream results into clear documents and saving "
             "them to the requested path.", ("write_file", "read_file")),
        Role("generalist", _COMMON, ("read_file", "list_dir", "calculator")),
    ]
}


def effective_tools(role: Role, task_tools: tuple[str, ...] | None, available: list[str]) -> list[str]:
    allowed = [t for t in role.tools if t in available]
    if task_tools is not None:
        allowed = [t for t in allowed if t in task_tools]
    return allowed
