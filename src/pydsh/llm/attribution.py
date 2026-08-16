"""App-attribution vocabulary for provider requests.

Ported from deepseek-harness ``packages/llm/llm/src/attribution.ts`` (MIT).

Centralizes the non-secret product identity every provider request sends as
``User-Agent``, keeping adapters from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata


def _package_version() -> str:
    try:
        return metadata.version("deepseek-harness")
    except metadata.PackageNotFoundError:
        return "0.0.0"


@dataclass(frozen=True)
class AppIdentity:
    """Static public application identity sent to LLM providers.

    Every field is a public product fact, safe on every request: no secrets,
    local paths, session ids, prompt text, or per-user identifiers belong
    here, and nothing per-request may influence the values.
    """

    #: ``User-Agent`` product token (lowercase, hyphenated).
    product: str
    #: Product version; sourced from package metadata, never hand-copied.
    version: str
    #: Repository home URL of the app, used as the ``User-Agent`` comment.
    url: str


#: The harness's own identity: the default every adapter sends.
APP_IDENTITY = AppIdentity(
    product="deepseek-harness",
    version=_package_version(),
    url="https://github.com/deepseek-ai/deepseek-harness",
)


def user_agent(identity: AppIdentity = APP_IDENTITY) -> str:
    """The standard ``User-Agent`` value: ``product/version (+url)``."""
    return f"{identity.product}/{identity.version} (+{identity.url})"


def attribution_headers(identity: AppIdentity = APP_IDENTITY) -> dict[str, str]:
    """Build the attribution headers an adapter must send on every provider
    request. Omission falls back to :data:`APP_IDENTITY`; nothing can suppress
    attribution entirely."""
    return {"user-agent": user_agent(identity)}
