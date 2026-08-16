"""Vocabulary for the filesystem Service Definition (``ctx.fs``).

Ported from deepseek-harness ``packages/fs/fs/src/types.ts`` (MIT).

Python mappings:

- The branded opaque ids ``FsTargetKey`` / ``FsVersion`` are ``NewType``
  wrappers over ``str``; calling them performs the (validation-free) branding.
- ``AbortSignal`` maps to :class:`asyncio.Event`: a *set* event means aborted.
- ``FsError`` carries a stable :data:`FsErrorCode`; the TS class extends
  ``HarnessError`` from ``dsh-llm``, which is not ported yet, so it extends
  :class:`Exception` directly here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, NewType, TypeAlias

#: Python mapping of the TS ``AbortSignal``: a set event means aborted.
AbortSignal: TypeAlias = asyncio.Event

#: Opaque key for stale guards and target lookup. The local backend uses a
#: realpath-like string; a remote backend might use a workspace URI or file
#: id. Consumers MUST NOT parse it or assume it is a local absolute path.
FsTargetKey = NewType('FsTargetKey', str)

#: Opaque file-version token — the freshness token a write/edit guards
#: against. The local backend derives it from high-resolution stat identity
#: and freshness fields; consumers may display related metadata but MUST NOT
#: interpret this token.
FsVersion = NewType('FsVersion', str)


@dataclass(frozen=True)
class FsObservationPresent:
    """A target was observed to exist at ``version``."""

    version: FsVersion
    kind: Literal['present'] = 'present'


@dataclass(frozen=True)
class FsObservationAbsent:
    """A target was authoritatively observed to be absent."""

    kind: Literal['absent'] = 'absent'


#: One authoritative observation of a target. A present observation carries
#: the version used by guarded replacement; an absent observation authorizes
#: only a guarded create, never an edit.
FsObservation: TypeAlias = FsObservationPresent | FsObservationAbsent


@dataclass(frozen=True)
class FsTarget:
    """A path resolved by a backend into a stable identity.

    ``resolve()`` produces this; every other operation takes it.
    """

    #: Opaque key for stale guards and target lookup.
    target_key: FsTargetKey
    #: Path for model/UI-facing output. May be a local absolute path,
    #: workspace-relative path, or remote URI depending on the backend.
    display_path: str


@dataclass(frozen=True)
class FsInfo:
    """Metadata about a target — what :meth:`FileSystem.stat` returns.

    Lets the policy layer reject directories/special files before reading and
    choose ``read_text`` vs ``stream_text`` from ``size`` without probing by
    failure. ``version`` is the freshness token. ``None`` from ``stat`` means
    the target is absent.
    """

    #: Opaque freshness token of the target right now.
    version: FsVersion
    #: Whether the target is a regular file, a directory, or something else.
    type: Literal['file', 'directory', 'other']
    #: Byte size of a regular file, when the backend can report it.
    size: int | None = None


@dataclass(frozen=True)
class FsPathInfo:
    """Metadata about a path without following a final symlink component.

    Unlike :class:`FsInfo`, this path-level probe can report ``symlink`` so
    consumers with trust-boundary rules can reject repository-owned links
    before resolving a target.
    """

    #: Opaque freshness token of the path entry right now.
    version: FsVersion
    #: Whether the path entry is a regular file, directory, symlink, or other.
    type: Literal['file', 'directory', 'symlink', 'other']
    #: Byte size of the path entry, when the backend can report it.
    size: int | None = None


@dataclass(frozen=True)
class FsDirEntry:
    """One direct child returned by :meth:`FileSystem.list_dir`.

    Listing returns metadata and resolved targets only; it must not read file
    contents.
    """

    #: Basename of the child inside the listed directory.
    name: str
    #: Whether the child is a regular file, a directory, or something else.
    type: Literal['file', 'directory', 'other']
    #: Resolved child target for follow-up operations.
    target: FsTarget
    #: Opaque freshness token when the backend can report metadata cheaply.
    version: FsVersion | None = None
    #: Byte size of a regular file, when the backend can report it.
    size: int | None = None


@dataclass(frozen=True)
class FsCreateIfAbsent:
    """Guarded write intent: create only when absent.

    Rejects an existing target with ``FS_NOT_OBSERVED``.
    """

    kind: Literal['createIfAbsent'] = 'createIfAbsent'


@dataclass(frozen=True)
class FsReplaceIfVersion:
    """Guarded write intent: replace only at the observed version.

    Rejects absence or mismatch with ``FS_STALE_VERSION``.
    """

    version: FsVersion
    kind: Literal['replaceIfVersion'] = 'replaceIfVersion'


#: Guarded write intent. Omitting the intent from ``write_text`` means
#: unconditional create-or-overwrite, not a third union arm.
FsWriteIntent: TypeAlias = FsCreateIfAbsent | FsReplaceIfVersion


@dataclass(frozen=True)
class FsWriteOutcome:
    """Outcome of a full-file write."""

    #: Whether the write created a new file or replaced an existing one.
    operation: Literal['create', 'update']
    #: Opaque version of the file after the write.
    version: FsVersion
    #: The file's content BEFORE the write, or ``None`` when the file did not
    #: exist (a create) or the backend declined a contextual basis (for
    #: example, a binary/non-UTF-8 prior file or either overwrite side
    #: reaching its exclusive limit). LF-normalized storage text (the diff
    #: basis), never a diff.
    before: str | None
    #: The file's content AFTER the write, LF-normalized to share ``before``'s
    #: diff basis.
    after: str


@dataclass(frozen=True)
class FsEditRequest:
    """A literal-replacement edit request."""

    #: Literal non-empty text to replace. Must match exactly (after
    #: line-ending normalization).
    old_string: str
    #: Literal replacement text. An empty string deletes the matched text.
    new_string: str
    #: Replace every match instead of requiring exactly one.
    replace_all: bool


@dataclass(frozen=True)
class FsEditOutcome:
    """Outcome of a literal edit."""

    #: Opaque version of the file after the edit.
    version: FsVersion
    #: The file's content BEFORE the edit. Raw storage text (LF-normalized by
    #: the backend), never a diff — a consumer computes the result-time
    #: contextual diff (the applied hunk with context) from
    #: ``before``/``after``.
    before: str
    #: The file's content AFTER the edit.
    after: str


#: Stable, machine-routable codes for filesystem failures. Carried on
#: :class:`FsError` so retry/permission/UI layers can branch without parsing
#: messages.
FsErrorCode: TypeAlias = Literal[
    'FS_NOT_FOUND',
    'FS_NOT_DIRECTORY',
    'FS_NOT_TEXT',
    'FS_NOT_REGULAR_FILE',
    'FS_TOO_LARGE',
    'FS_PERMISSION_DENIED',
    'FS_SANDBOX_DENIED',
    'FS_IO_ERROR',
    'FS_STALE_VERSION',
    'FS_NOT_OBSERVED',
    'FS_AMBIGUOUS_EDIT',
    'FS_EDIT_NOT_FOUND',
    'FS_ABORTED',
]


class FsError(Exception):
    """Typed filesystem error carrying a stable :data:`FsErrorCode`.

    ``pydsh.fs`` owns this vocabulary so backends and the policy layer raise
    the same codes instead of each inventing message strings. ``cause``
    chains the originating error like the TS ``ErrorOptions.cause``.
    """

    def __init__(
        self,
        message: str,
        code: FsErrorCode,
        *,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code: FsErrorCode = code
        if cause is not None:
            self.__cause__ = cause
