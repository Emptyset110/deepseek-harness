"""Shared agent-loop scheduler defaults.

Ported from deepseek-harness ``packages/core/agent-loop/src/constants.ts``
(MIT).
"""

#: Default maximum in-flight parallel-safe calls per agent step.
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 10
