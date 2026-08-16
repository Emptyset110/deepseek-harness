"""Base service class and service lifecycle conventions.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/service.ts``). The well-known symbols map onto Python
conventions: ``Service.init`` → a ``cordis_init()`` method run after
construction of class plugins, ``Service.check`` → a ``check()`` predicate
attribute passed to ``ctx.provide()``, ``Service.invoke`` → ``__call__``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from .utils import Tracker

if TYPE_CHECKING:
    from .context import Context


class Service:
    """Base class for services that expose a named API on ``ctx``.

    Subclasses call ``super().__init__(ctx, name)``. The service is
    registered immediately and is automatically removed with the owning
    fiber. A subclass may set the class attribute ``provide`` as the default
    name and define ``check(self) -> bool`` as the availability predicate.
    """

    _cordis_tracker = Tracker(property='ctx')

    def __init__(self, ctx: Context, name: str | None = None) -> None:
        if name is None:
            name = getattr(type(self), 'provide', None)
        if name is None:
            raise TypeError('service name is required')
        self.ctx = ctx
        self.name = name
        self._cordis_tracker = Tracker(associate=name, property='ctx')
        ctx.reflect.provide(name, self, getattr(self, 'check', None))

    def filter(self, ctx: Context) -> bool:
        """Scope filter matching contexts in this service's isolation scope."""
        return bool(
            ctx._cordis_isolate.get(self.name)
            == self.ctx._cordis_isolate.get(self.name)
        )

    def extend(self, props: dict[str, Any] | None = None) -> Any:
        """Derive an extended service instance shadowing this one."""
        new = copy.copy(self)
        if props:
            new.__dict__.update(props)
        return new

    def resolve_config(self, base: Any = None, head: Any = None) -> Any:
        """Merge intercept config from ancestors with base and head values.

        Entries added closer to the root apply first; ``base`` is prepended
        and ``head`` appended. Uses ``Config.merge`` when the service
        declares one, otherwise a shallow dict merge.
        """
        intercept = self.ctx._cordis_intercept
        configs: list[Any] = []
        node = intercept
        while node is not None and self.name in node:
            if self.name in node.own:
                configs.insert(0, node.own[self.name])
            node = node.parent
        if base is not None:
            configs.insert(0, base)
        if head is not None:
            configs.append(head)
        config_cls = getattr(self, 'Config', None)
        merge = getattr(config_cls, 'merge', None)
        if merge is not None:
            return merge(*configs)
        result: dict[str, Any] = {}
        for config in configs:
            result.update(config)
        return result
