"""Behavior tests for the logger service.

Ported-semantics tests for the cordis Python port (MIT, cordiverse;
vendored in deepseek-harness).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

from pycordis import Context, Logger, LoggerLevel


async def test_logger_records_into_buffer() -> None:
    ctx = Context()
    ctx.logger.info('hello %s', 'world')
    message = ctx.logger.buffer[-1]
    assert message.type == 'info'
    assert message.level == LoggerLevel.INFO
    assert message.name == 'root'


async def test_logger_format() -> None:
    ctx = Context()
    # the vendored level order is ERROR < INFO < WARN < DEBUG and the default
    # exporter threshold is INFO, so warn/debug messages are skipped by default
    ctx.logger('test').info('value: %d extra', 3.9, 'tail')
    message = ctx.logger.buffer[-1]
    assert message.name == 'test'
    formatted = Logger.format(SimpleNamespace(), message)
    assert formatted == 'value: 3 extra tail'


async def test_logger_name_from_fiber() -> None:
    ctx = Context()

    def myPlugin(c: Context, config: object) -> None:
        c.logger.info('from plugin')

    fiber = ctx.plugin(myPlugin)
    await fiber
    assert ctx.logger.buffer[-1].name == 'my-plugin'


async def test_logger_error_unwraps_exception_group() -> None:
    ctx = Context()
    group = ExceptionGroup('group', [ValueError('one'), ValueError('two')])
    ctx.logger.error(group)
    messages = ctx.logger.buffer
    assert messages[-2].args[0].args[0] == 'one'
    assert messages[-1].args[0].args[0] == 'two'


async def test_exporter_level_filtering() -> None:
    ctx = Context()
    received: list[object] = []

    class Exporter:
        # levels cap message verbosity: levels above the target are skipped
        levels: ClassVar[dict[str, int]] = {'default': LoggerLevel.INFO}

        def export(self, message: object) -> None:
            received.append(message)

    ctx.logger.exporter(Exporter())
    ctx.logger.info('kept')
    ctx.logger.warn('skipped')
    assert [m.args[0] for m in received] == ['kept']
