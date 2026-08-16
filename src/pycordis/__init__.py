"""Pure-Python port of the Cordis framework kernel.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src``, upstream cordiverse/cordis @ 56b3d4f + dsh local
modifications).
"""

from .context import Context
from .events import EventsService, Hook, is_bailed
from .fiber import (
    CordisError,
    EffectDisposer,
    Fiber,
    FiberState,
    ValidationError,
    resolve_config,
)
from .logger import Logger, LoggerLevel, LoggerService, Message, default_formatters
from .reflect import Impl, Label, ReflectService
from .registry import PluginRuntime, RegistryService, resolve_inject
from .service import Service
from .utils import ChainMap, DisposableList, ServiceView, Tracker, rebind, symbols

__all__ = [
    'ChainMap',
    'Context',
    'CordisError',
    'DisposableList',
    'EffectDisposer',
    'EventsService',
    'Fiber',
    'FiberState',
    'Hook',
    'Impl',
    'Label',
    'Logger',
    'LoggerLevel',
    'LoggerService',
    'Message',
    'PluginRuntime',
    'ReflectService',
    'RegistryService',
    'Service',
    'ServiceView',
    'Tracker',
    'ValidationError',
    'default_formatters',
    'is_bailed',
    'rebind',
    'resolve_config',
    'resolve_inject',
    'symbols',
]
