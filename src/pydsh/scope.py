"""Scoped-context primitives: the single pydsh port of dsh-scope.

Ported from deepseek-harness ``packages/core/scope`` (MIT). This module is
the one home of the scope machinery, merged from the two earlier parallel
stand-ins (``pydsh.system_prompt._scope`` and the carrier subset of
``pydsh.agent.dispatch``): identity-compared scope keys with parent links,
scope minting over a no-op backing plugin, scope-filtered event carriers
(``scope_target``), and the insertion-ordered entry stores
(``NamedEntries`` / ``AnonymousEntries`` / ``ScopedLayers``). The TypeScript
``isScopeCarrier`` / ``carrierKeyOf`` introspection helpers are not ported:
the carrier exposes its routing key as the public ``key`` attribute.

Python mappings:

- Scope keys are plain identity-compared objects (create them with
  ``object()``); the parent map holds strong references where TypeScript
  uses a ``WeakMap``.
- A context's scope tag is stored as the ``dsh_scope`` meta entry written by
  ``create_scope()`` (``kScope`` in TypeScript); ``scope_of()`` walks the
  ``extend()`` parent chain, matching TypeScript's prototype-chain read.
- The event carrier exposes its filter as the ``_cordis_filter`` attribute
  read by the event dispatcher (``Context[symbols.filter]`` in TypeScript).
  A base subject's own filter is read from its ``_cordis_filter`` attribute
  (plain-object subjects) or its ``filter`` method (Cordis services).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from pycordis import Context, EffectDisposer

#: An opaque, identity-compared scope key.
ScopeKey = object

#: Context meta key written by ``create_scope()`` (``kScope`` in TypeScript).
SCOPE_ATTR = 'dsh_scope'

# The enclosing scope of each key. One relation powers both directions of
# scope nesting: registration views inherit DOWN the chain, and event
# admission extends UP it. Strong references where TypeScript uses a WeakMap.
_scope_parents: dict[ScopeKey, ScopeKey] = {}


def scope_of(ctx: Context) -> ScopeKey | None:
    """Read the nearest scope tag inherited by a context.

    :param ctx: context to inspect.
    :return: its scope key, or ``None`` for an unscoped context.
    """
    node: Context | None = ctx
    while node is not None:
        if SCOPE_ATTR in node.__dict__:
            return cast(ScopeKey, node.__dict__[SCOPE_ATTR])
        node = node.__dict__.get('_parent')
    return None


def _link_scope_parent(key: ScopeKey, parent: ScopeKey) -> None:
    """Cycle-checked write shared by the bind and every rebind."""
    cursor: ScopeKey | None = parent
    while cursor is not None:
        if cursor is key:
            raise ValueError('dsh-scope: scope parent link would form a cycle')
        cursor = _scope_parents.get(cursor)
    _scope_parents[key] = parent


class ScopeParentBinding(Protocol):
    """The privileged handle to move one scope key's parent link."""

    def rebind(self, parent: ScopeKey) -> None:
        """Re-link the bound key to a different parent, with cycle checking."""
        ...


def bind_scope_parent(key: ScopeKey, parent: ScopeKey) -> ScopeParentBinding:
    """Bind ``parent`` as ``key``'s enclosing scope, once.

    A key that already has a parent raises: there is no open re-link path,
    so a scope's ancestry cannot be moved by anyone but the original binder,
    who alone receives the binding.

    :param key: the child scope key.
    :param parent: its enclosing scope key.
    :return: the binding that alone may re-link this key.
    """
    if key in _scope_parents:
        raise ValueError(
            'dsh-scope: scope key is already bound to a parent; re-linking '
            'requires the binding returned by the original bind'
        )
    _link_scope_parent(key, parent)

    class _Binding:
        def rebind(self, next_parent: ScopeKey) -> None:
            _link_scope_parent(key, next_parent)

    return _Binding()


def scope_parent_of(key: ScopeKey) -> ScopeKey | None:
    """Read one key's enclosing scope, or ``None`` for a root scope."""
    return _scope_parents.get(key)


def scope_chain_of(key: ScopeKey | None) -> list[ScopeKey]:
    """The chain from a key to its root ancestor, nearest-first."""
    chain: list[ScopeKey] = []
    cursor = key
    while cursor is not None:
        chain.append(cursor)
        cursor = _scope_parents.get(cursor)
    return chain


def _scope_plugin(ctx: Context, config: Any = None) -> None:
    """Shared no-op plugin used as the backing scope fiber."""


@dataclass
class Scope:
    """A minted registration scope and its disposal boundaries."""

    #: Context through which scope-owned registrations are made.
    ctx: Context
    #: Exact Cordis disposer of the backing fiber.
    raw_dispose: Callable[[], Any]
    #: Dispose every scope-owned registration; racing calls await the same
    #: completion (the first call starts teardown and returns the shared
    #: awaitable).
    dispose: Callable[[], Awaitable[None]]


def create_scope(
    ctx: Context,
    key: ScopeKey,
    parent: ScopeKey | None = None,
) -> Scope:
    """Mint a scope under ``ctx``.

    The scoped context inherits the minting plugin's dependency API and owns
    every registration made through it.

    :param ctx: active context whose dependency API the scope inherits.
    :param key: opaque identity used for listener routing.
    :param parent: optional enclosing scope bound before the scope is usable.
    :return: the scoped context and disposal boundaries.
    """
    if parent is not None:
        bind_scope_parent(key, parent)
    fiber = ctx.plugin(_scope_plugin)
    scoped = fiber.ctx.extend({SCOPE_ATTR: key})

    async def quiesce() -> None:
        # Follow the fiber through asynchronous teardown even if its raw
        # disposer was already claimed.
        result = fiber.dispose()
        if inspect.isawaitable(result):
            await result
        while fiber.inertia is not None:
            await fiber.inertia

    disposing: asyncio.Task[None] | None = None

    def dispose() -> asyncio.Task[None]:
        nonlocal disposing
        if disposing is None:
            disposing = asyncio.ensure_future(quiesce())
        return disposing

    return Scope(ctx=scoped, raw_dispose=fiber.dispose, dispose=dispose)


A = TypeVar('A', bound=object)


class ScopeCarrier(Generic[A]):
    """Routing-only event receiver built by ``scope_target()`` (TS ``Scoped<T>``).

    The dispatcher reads the ``_cordis_filter`` attribute and prepends the
    carrier as the listener's first argument (JavaScript ``this``); event
    payloads carry the real subject. The carrier does not expose the
    subject's properties; the subject type is recorded only for dispatch
    checking.
    """

    __slots__ = ('_base_filter', '_key')

    def __init__(self, base: A, key: ScopeKey | None) -> None:
        base_filter: Any = getattr(base, Context.filter, None)
        if base_filter is None:
            base_filter = getattr(base, 'filter', None)
        self._base_filter: Callable[[Context], bool] | None = (
            base_filter if callable(base_filter) else None
        )
        self._key = key

    @property
    def key(self) -> ScopeKey | None:
        """The routed scope identity this carrier dispatches to."""
        return self._key

    def _cordis_filter(self, ctx: Context) -> bool:
        if self._base_filter is not None and not self._base_filter(ctx):
            return False
        tag = scope_of(ctx)
        # An untagged listener context is admitted globally; a tagged one is
        # admitted for a matching key or any of its ancestors, so a listener
        # owned by an enclosing scope receives every descendant scope's
        # events. A tag BELOW the dispatch key stays excluded — events flow
        # up the chain, never down.
        if tag is None:
            return True
        cursor = self._key
        while cursor is not None:
            if cursor is tag:
                return True
            cursor = _scope_parents.get(cursor)
        return False


def scope_target(base: A, key: ScopeKey | None) -> ScopeCarrier[A]:
    """Build an opaque receiver preserving the base filter.

    Untagged listeners are admitted globally; tagged listeners are admitted
    for a matching key or any of its ancestors (``bind_scope_parent``), so a
    listener owned by an enclosing scope receives every descendant scope's
    events. A tag BELOW the dispatch key stays excluded — events flow up the
    chain, never down.

    :param base: subject or service whose existing Cordis filter is
        preserved (its ``_cordis_filter`` attribute or ``filter`` method).
    :param key: routed scope identity, or ``None`` for an unscoped subject.
    :return: a carrier whose subject remains available only through arguments.
    """
    return ScopeCarrier(base, key)


V = TypeVar('V')
L = TypeVar('L', bound='ScopeLayer')


class ScopeLayer(Protocol):
    """One scope's aggregate contribution to a registry."""

    def is_empty(self) -> bool:
        """Whether every table in this layer is empty."""
        ...


class NamedEntries(Generic[V]):
    """Insertion-ordered named entries with caller-owned duplicate diagnostics.

    Values are borrowed. Each successful insertion returns an idempotent
    undo for that exact entry; draining the table detaches it from later
    insertions.
    """

    def __init__(self, duplicate_error: Callable[[str], Exception]) -> None:
        self._duplicate_error = duplicate_error
        self._data: dict[str, V] = {}

    def insert(self, name: str, value: V) -> Callable[[], None]:
        """Insert one unique name; returns an idempotent undo for it."""
        data = self._data
        if name in data:
            raise self._duplicate_error(name)
        data[name] = value
        active = True

        def undo() -> None:
            nonlocal active
            if not active:
                return
            active = False
            del data[name]
            if not data and data is self._data:
                self._data = {}

        return undo

    def get(self, name: str) -> V | None:
        """Read one named value, or ``None`` when absent."""
        return self._data.get(name)

    def has(self, name: str) -> bool:
        """Test one name for membership."""
        return name in self._data

    def keys(self) -> Iterator[str]:
        """Iterate live names in insertion order."""
        return iter(self._data.keys())

    def entries(self) -> Iterator[tuple[str, V]]:
        """Iterate live entries in insertion order."""
        return iter(self._data.items())

    def values(self) -> Iterator[V]:
        """Iterate live values in insertion order."""
        return iter(self._data.values())

    def is_empty(self) -> bool:
        """Test whether this table has no entries."""
        return not self._data


class AnonymousEntries(Generic[V]):
    """Insertion-ordered anonymous entries with independent registration identity.

    Equal values remain separate registrations. Appending returns an
    idempotent undo for that exact append; draining the table detaches it
    from later appends.
    """

    def __init__(self) -> None:
        self._data: dict[object, V] = {}

    def append(self, value: V) -> Callable[[], None]:
        """Append one independently owned value; returns its idempotent undo."""
        data = self._data
        key = object()
        data[key] = value
        active = True

        def undo() -> None:
            nonlocal active
            if not active:
                return
            active = False
            del data[key]
            if not data and data is self._data:
                self._data = {}

        return undo

    def values(self) -> Iterator[V]:
        """Iterate live values in insertion order."""
        return iter(self._data.values())

    def is_empty(self) -> bool:
        """Test whether this table has no entries."""
        return not self._data


class ScopedLayers(Generic[L]):
    """Own the global and exact-scope layers for one registry.

    Reads never create scoped layers. Registrations derive both visibility
    and effect ownership from the supplied Cordis context, collect undo
    before notification, and reclaim only a completely empty aggregate layer.
    """

    def __init__(
        self,
        create_layer: Callable[[ScopeKey | None], L],
        on_change: Callable[[], None],
    ) -> None:
        self._create_layer = create_layer
        self._on_change = on_change
        #: The eagerly constructed context-global layer.
        self.global_: L = create_layer(None)
        self._scoped: dict[ScopeKey, L] = {}

    def peek(self, scope: ScopeKey | None) -> L | None:
        """Read an existing exact-scope overlay without creating one.

        Deliberately chain-blind: callers addressing one scope's OWN
        contributions must not silently pick up an ancestor's — use
        ``chain_layers()`` where inheritance is the point.
        """
        if scope is None:
            return None
        return self._scoped.get(scope)

    def chain_layers(self, scope: ScopeKey | None) -> list[L]:
        """Existing overlays along the scope's parent chain.

        Farthest ancestor first and the exact scope last, so a caller
        layering them in order gives the nearest scope the final word.
        """
        layers: list[L] = []
        for key in reversed(scope_chain_of(scope)):
            layer = self._scoped.get(key)
            if layer is not None:
                layers.append(layer)
        return layers

    def merge(
        self,
        scope: ScopeKey | None,
        pick: Callable[[L], NamedEntries[V]],
    ) -> dict[str, V]:
        """Materialize global named entries followed by scope-chain shadows.

        Farthest ancestor first, so the nearest scope's entry wins a name.
        """
        merged = dict(pick(self.global_).entries())
        for layer in self.chain_layers(scope):
            merged.update(pick(layer).entries())
        return merged

    def effect(
        self,
        ctx: Context,
        action: Callable[[L], Callable[[], None]],
        label: str,
        notify: bool = True,
    ) -> EffectDisposer:
        """Attach one synchronous layer mutation to its registration context.

        :param ctx: context determining scope visibility and effect ownership.
        :param action: atomic mutation returning its synchronous undo.
        :param label: Cordis effect label.
        :param notify: whether to fire the change notification.
        :return: the exact disposer returned by ``ctx.effect()``.
        """
        scope = scope_of(ctx)

        def run() -> Iterator[Callable[[], None]]:
            created = False
            if scope is None:
                layer = self.global_
            else:
                existing = self._scoped.get(scope)
                if existing is None:
                    layer = self._create_layer(scope)
                    self._scoped[scope] = layer
                    created = True
                else:
                    layer = existing

            try:
                undo = action(layer)
            except BaseException:
                if scope is not None and created and layer.is_empty():
                    del self._scoped[scope]
                raise

            def dispose() -> None:
                undo()
                if scope is not None and layer.is_empty():
                    self._scoped.pop(scope, None)
                if notify:
                    self._on_change()

            yield dispose
            if notify:
                self._on_change()

        return ctx.effect(run, label)
