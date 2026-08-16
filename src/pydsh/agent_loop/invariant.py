"""Package-owned request-reconstruction invariant for loop-built LLM calls.

Ported from deepseek-harness
``packages/core/agent-loop/src/invariant.ts`` (MIT).

The TypeScript companion registers through the ``dsh-invariants`` service,
which the Python rewrite does not have; ``install()`` here takes the failure
reporter directly and keeps the same waterfall placement (global, prepended),
mirroring ``pydsh.system_prompt.invariant``.

The ``Object.isFrozen`` checks have no Python equivalent (harness data dicts
are read-only by convention) and are not ported; the durable reconstruction
checks — live session id, ``step/start`` presence, folded request header, and
log-derived message equality — are fully ported.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from pycordis import Context, EffectDisposer
from pydsh.llm import GenerateOptions, is_agent_loop_request
from pydsh.session import SessionId, fold_request_header

#: Invariant failure reporter: one call per violated invariant.
InvariantFailure = Callable[[str], None]

#: Cordis companion plugin name (TS ``name = 'agent-loop-invariant'``).
NAME = 'agent-loop-invariant'


def install(ctx: Context, fail: InvariantFailure) -> EffectDisposer:
    """Install request-reconstruction validation around ``llm/stream``.

    The listener runs prepended and global, so a short-circuiting replay
    listener cannot silence the check.

    :param ctx: context owning the listener registration (needs ``sessions``).
    :param fail: reporter called once per violated invariant.
    :return: the listener's disposer.
    """
    def listener(
        carrier: Any, options: GenerateOptions, next_: Callable[[], Any]
    ) -> Any:
        if not is_agent_loop_request(options):
            return next_()
        if options.get('sessionId') is None:
            fail('a loop-built request must carry a session id')
        session = ctx.get('sessions').get(SessionId(options.get('sessionId') or ''))
        if session is None:
            fail(
                'a loop-built request must carry a live session id, got'
                f' "{options.get("sessionId")}"'
            )

        events = session.events if session is not None else ()
        if not any(event['type'] == 'step/start' for event in events):
            # Mirrors TS `return fail(...)`: the waterfall short-circuits.
            return fail(
                'a loop-built request with no step/start in its session log'
            )
        header = fold_request_header(events)
        if header is None:
            return fail(
                'a loop-built request with no request/header event in its'
                ' session log'
            )
        expected = session.derive_messages()
        if json.dumps(options['messages'], sort_keys=True) != json.dumps(
            cast(Any, expected), sort_keys=True
        ):
            fail(
                f'llm request for session "{session.id}" diverges from the'
                ' dispatch-time durable derivation (log-reconstruction desync)'
            )

        header_config = header['config']
        header_matches = (
            options.get('model') == header_config.get('model')
            and options.get('system') == header.get('system')
            and options.get('temperature') == header_config.get('temperature')
            and options.get('maxTokens') == header_config.get('maxTokens')
            and json.dumps(options.get('stop')) == json.dumps(header_config.get('stop'))
            and json.dumps(options.get('tools') or [])
            == json.dumps(header.get('tools') or [])
        )
        if not header_matches:
            fail(
                f'llm request for session "{session.id}" diverges from the'
                ' folded request header'
            )
        return next_()

    return cast(
        EffectDisposer,
        ctx.on('llm/stream', listener, {'global': True, 'prepend': True}),
    )
