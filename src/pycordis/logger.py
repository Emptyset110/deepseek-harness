"""Logger facade, logger service, message, exporter, and formatting types.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/logger.ts``). The callable-service proxy machinery maps
onto ``LoggerService.__call__``.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from .utils import Tracker

if TYPE_CHECKING:
    from .context import Context
    from .fiber import Fiber

LoggerType = str  # 'error' | 'info' | 'warn' | 'debug'
Formatter = Any  # (value, exporter, message) -> Any


class LoggerLevel(IntEnum):
    """Numeric severity used when exporters decide whether to emit a message."""

    ERROR = 0
    INFO = 1
    WARN = 2
    DEBUG = 3


@dataclass
class Message:
    """Structured log record delivered to exporters."""

    sn: int
    ts: float
    name: str
    type: LoggerType
    level: int
    args: list[Any]
    fiber: weakref.ReferenceType[Fiber] | None = None


def _to_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _to_int(value: Any) -> Any:
    number = _to_number(value)
    try:
        return int(number)  # truncates toward zero like Math.trunc
    except (ValueError, OverflowError):
        return number  # NaN/inf mirror the JavaScript output


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _color_name(value: Any, exporter: Any, message: Message) -> str:
    return Logger.color(
        exporter, Logger.code(message.name, getattr(exporter, 'colors', None)), value
    )


default_formatters: dict[str, Formatter] = {
    's': lambda value, *_: str(value),
    'd': lambda value, *_: _to_int(value),
    'i': lambda value, *_: _to_int(value),
    'f': lambda value, *_: _to_number(value),
    'o': lambda value, *_: _json(value),
    'O': lambda value, *_: _json(value),
    'c': lambda *_: '',
    'C': _color_name,
}


class Logger:
    """Logger facade for one named subsystem."""

    def __init__(self, options: dict[str, Any], service: LoggerService) -> None:
        self.name: str = options['name']
        self.meta: dict[str, Any] = options.get('meta') or {}
        self.level: int | None = options.get('level')
        self.service = service

    @staticmethod
    def color(exporter: Any, code: int, value: Any, decoration: str = '') -> str:
        colors = getattr(exporter, 'colors', None)
        if not colors:
            return str(value)
        suffix = decoration if colors >= 2 else ''
        prefix = f'3{code}' if code < 8 else f'38;5;{code}'
        return f'\x1b[{prefix}{suffix}m{value}\x1b[0m'

    @staticmethod
    def code(name: str, level: int | None = None) -> int:
        hash_ = 0
        for char in name:
            hash_ = ((hash_ << 3) - hash_) + ord(char) + 13
            hash_ &= 0xFFFFFFFF
            if hash_ >= 0x80000000:
                hash_ -= 0x100000000
        colors: list[int] = []
        if level:
            colors = c256 if level >= 2 else c16
        if not colors:
            # JavaScript indexes the empty palette to undefined; the caller
            # only reaches this when coloring is disabled.
            return 0
        return colors[abs(hash_) % len(colors)]

    @staticmethod
    def format(exporter: Any, message: Message) -> str:
        args = list(message.args)
        if args and isinstance(args[0], BaseException):
            error = args[0]
            args[0] = ''.join(traceback.format_exception(error)).rstrip()
            args.insert(0, '%s')
        elif not args or not isinstance(args[0], str):
            args.insert(0, '%o')

        format_str = str(args.pop(0))
        formatters = getattr(exporter, 'formatters', None) or {}

        def replace(match: re.Match[str]) -> str:
            char = match.group(1)
            if match.group(0) == '%%':
                return '%'
            formatter = formatters.get(char) or default_formatters.get(char)
            if callable(formatter):
                value = args.pop(0) if args else None
                return str(formatter(value, exporter, message))
            return match.group(0)

        format_str = re.sub(r'%([a-zA-Z%])', replace, format_str)

        o_formatter = formatters.get('o') or default_formatters['o']
        for arg in args:
            if arg is not None and not isinstance(arg, (str, bytes, int, float, bool)):
                arg = o_formatter(arg, exporter, message)
            format_str += ' ' + str(arg)

        max_length = getattr(exporter, 'maxLength', None) or 10240
        return '\n'.join(
            line[:max_length] + ('...' if len(line) > max_length else '')
            for line in re.split(r'\r?\n', format_str)
        )

    def error(self, *args: Any) -> None:
        self._log('error', LoggerLevel.ERROR, args)

    def info(self, *args: Any) -> None:
        self._log('info', LoggerLevel.INFO, args)

    def warn(self, *args: Any) -> None:
        self._log('warn', LoggerLevel.WARN, args)

    def debug(self, *args: Any) -> None:
        self._log('debug', LoggerLevel.DEBUG, args)

    def _log(self, type_: LoggerType, level: int, args: tuple[Any, ...]) -> None:
        if len(args) == 1 and isinstance(args[0], BaseException):
            error = args[0]
            if error.__cause__ is not None:
                self._log(type_, level, (error.__cause__,))
            elif isinstance(error, BaseExceptionGroup):
                for inner in error.exceptions:
                    self._log(type_, level, (inner,))
                return

        self.service._sn_message += 1
        sn = self.service._sn_message
        ts = time.time() * 1000
        for exporter in list(self.service.exporters.values()):
            levels = getattr(exporter, 'levels', None) or {}
            target_level = levels.get(self.name)
            if target_level is None:
                target_level = levels.get('default')
            if target_level is None:
                target_level = self.level
            if target_level is None:
                target_level = LoggerLevel.INFO
            if target_level < level:
                continue
            message = Message(
                sn=sn,
                ts=ts,
                type=type_,
                level=level,
                name=self.name,
                args=list(args),
                fiber=self.meta.get('fiber'),
            )
            exporter.export(message)


c16 = [6, 2, 3, 4, 5, 1]
c256 = [
    20, 21, 26, 27, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45, 56, 57, 62,
    63, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 92, 93, 98, 99, 112, 113,
    129, 134, 135, 148, 149, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 178, 179, 184, 185, 196, 197, 198, 199, 200,
    201, 202, 203, 204, 205, 206, 207, 208, 209, 214, 215, 220, 221,
]


def hyphenate(source: str) -> str:
    return re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', source).lower()


class LoggerService:
    """Built-in logging service.

    Call ``ctx.logger()`` to create a named logger, or call
    ``ctx.logger.info()`` directly to log with the current fiber-derived name.
    """

    _cordis_tracker = Tracker(property='ctx', no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.buffer_size = 1000
        self.buffer: list[Message] = []
        self._sn_message = 0
        self._sn_exporter = 0
        self.exporters: dict[int, Any] = {}

        service = self

        class BufferExporter:
            colors = 3

            def export(self, message: Message) -> None:
                service.buffer.append(message)
                if len(service.buffer) > service.buffer_size:
                    service.buffer[:] = service.buffer[-service.buffer_size:]

        self.exporter(BufferExporter())

    def exporter(self, exporter: Any) -> Any:
        """Register an exporter and dispose it with the current fiber."""
        def effect() -> Any:
            self._sn_exporter += 1
            self.exporters[self._sn_exporter] = exporter
            # Mirrors the vendored source: removal keys off the latest
            # exporter serial rather than the one captured at registration.
            return lambda: self.exporters.pop(self._sn_exporter, None)

        return self.ctx.effect(effect, 'ctx.logger.exporter()')

    def _resolve_config(self) -> dict[str, Any]:
        intercept = self.ctx._cordis_intercept
        configs: list[dict[str, Any]] = []
        node = intercept
        while node is not None and 'logger' in node:
            if 'logger' in node.own:
                configs.insert(0, node.own['logger'])
            node = node.parent
        result: dict[str, Any] = {}
        for config in configs:
            result.update(config)
        return result

    def __call__(self, name: str | None = None) -> Logger:
        config = self._resolve_config()
        fiber = self.ctx.fiber
        if name is None:
            name = config.get('name')
        if name is None:
            name = hyphenate(fiber.name)
        return Logger(
            {'name': name, 'level': config.get('level'),
             'meta': {'fiber': weakref.ref(fiber)}},
            self,
        )

    def error(self, *args: Any) -> None:
        self().error(*args)

    def info(self, *args: Any) -> None:
        self().info(*args)

    def warn(self, *args: Any) -> None:
        self().warn(*args)

    def debug(self, *args: Any) -> None:
        self().debug(*args)
