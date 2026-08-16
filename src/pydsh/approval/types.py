"""Wire-safe approval identifiers and outcome vocabulary.

Ported from deepseek-harness packages/interaction/user-approval (MIT),
``src/types.ts``. Free of cordis/service imports so any layer can consume
the vocabulary without loading the service module.

Python mapping: the TS branded string becomes :class:`typing.NewType`;
calling ``ApprovalRequestId(raw)`` brands a plain string, mirroring the TS
cast function.
"""

from __future__ import annotations

from typing import Literal, NewType, TypeAlias

ApprovalRequestId = NewType('ApprovalRequestId', str)
"""Pairs one ``approval/asked`` audit event with its ``approval/decided``.

Service-issued (one fresh id per ``ApprovalService.request`` call).
"""

ApprovalOutcome: TypeAlias = Literal[
    'allowed-once', 'rejected', 'cancelled', 'unavailable'
]
"""Closed approval outcomes: a one-shot grant, explicit rejection, withdrawn
request, or unavailable answerer. Callers fail closed on ``unavailable``."""
