"""Manage the process-level kernel selector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator
from contextlib import contextmanager

from phyai.kernel.types import ModelContext
from phyai.kernel.policy import Policy, load_policy, policy_from_mapping
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phyai.kernel.config import KernelConfig

# note(chenghua.wang): Single process can only hold single engine and single selector.
_selector: Selector | None = None


def get_kernel_selector() -> Selector:
    """Return the process selector, creating the default selector when needed."""

    global _selector
    if _selector is None:
        _selector = Selector(build_catalog())
    return _selector


@contextmanager
def kernel_selector_scope(
    selector: Selector | None = None,
) -> Generator[Selector | None]:
    """Temporarily install a selector and restore the previous value on exit."""

    global _selector
    saved = _selector
    _selector = selector
    try:
        yield selector
    finally:
        _selector = saved


def set_kernel_selector(selector: Selector | None) -> None:
    global _selector
    _selector = selector


def reset_kernel_selector() -> None:
    """Drop the installed selector so the next use rebuilds the default."""

    set_kernel_selector(None)


def resolve_policy(config: "KernelConfig", catalog: Catalog) -> Policy:
    """Load the configured policy and apply an explicit profile override."""

    policy = (
        policy_from_mapping(
            config.policy_config, catalog, source="KernelConfig.policy_config"
        )
        if config.policy_config is not None
        else load_policy(config.config_path, catalog)
    )
    if config.profile is not None and config.profile.lower() != policy.profile:
        policy = Policy(
            profile=config.profile,
            fallback=policy.fallback,
            rules=policy.rules,
            overrides=policy.overrides,
            source=policy.source,
        )

    return policy


def initialize_kernel_system(
    config: "KernelConfig | None" = None,
    *,
    device: object | None = None,
    model: ModelContext | None = None,
) -> Selector:
    """Install the engine's selector and return it."""

    from phyai.kernel.config import KernelConfig

    config = config or KernelConfig()
    catalog = build_catalog()
    policy = resolve_policy(config, catalog)
    benchmark = config.benchmark
    if benchmark is None and policy.profile == "autotune":
        # ``profile: autotune`` with no programmatic hook gets the engine's
        # default select-time benchmark, so PHYAI_KERNEL_PROFILE=autotune
        # works end-to-end from the environment alone.
        from phyai.kernel.benchmark import default_benchmark

        benchmark = default_benchmark(catalog)
    selector = Selector(
        catalog,
        policy,
        model=model,
        device=device,
        benchmark=benchmark,
        autotune_cache=config.autotune_cache,
    )
    set_kernel_selector(selector)
    return selector


__all__ = [
    "get_kernel_selector",
    "initialize_kernel_system",
    "kernel_selector_scope",
    "reset_kernel_selector",
    "resolve_policy",
    "set_kernel_selector",
]
