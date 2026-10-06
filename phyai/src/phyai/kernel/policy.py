"""Compile kernel policy YAML into typed predicates."""

from __future__ import annotations

import json
import hashlib
from typing import Any, Mapping, Iterable, Sequence, TYPE_CHECKING
from pathlib import Path
from dataclasses import field, dataclass

from phyai.kernel.facts import (
    GLOBAL_FACT_PATHS,
    OP_SCOPED_PREFIXES,
    suggest_name,
)
from phyai.kernel.facts import (
    attrs as attrs_ns,
)
from phyai.kernel.facts import (
    dtype as dtype_ns,
)
from phyai.kernel.facts import (
    shape as shape_ns,
)
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import (
    TRUE,
    Fact,
    FactKind,
    Predicate,
    all_of,
    any_of,
    none_of,
    predicate_from_literal,
)

if TYPE_CHECKING:
    from phyai.layers.quant.plan import QuantPlan

SCHEMA = "phyai.kernel/v1"

#: Accepted spellings of the schema marker.
_SCHEMA_VALUES = frozenset({SCHEMA, "v1", 1, "1"})

#: Reserved matcher combinators.
_COMBINATORS = frozenset({"any_of", "all_of", "none_of", "when"})

_RULE_FIELDS = frozenset({"id", "priority", "match", "prefer", "restrict_to", "params"})
_OVERRIDE_FIELDS = frozenset(
    {"id", "priority", "match", "use", "restrict_to", "params"}
)

_NAMESPACES = frozenset({"device", "quant", "model", "shape", "dtype", "attrs", "lib"})


class PolicyError(ValueError):
    """Raised when a kernel policy cannot be compiled."""


def _fact_for(path: str, spec_paths: Mapping[str, FactKind] | None) -> Fact:
    """Return the typed fact for a dotted path, or raise with a suggestion."""

    prefix, _, name = path.partition(".")

    if prefix in OP_SCOPED_PREFIXES:
        if spec_paths is not None and path not in spec_paths:
            raise PolicyError(f"unknown fact {path!r}{_suggest(path, spec_paths)}")
        kind = (
            spec_paths[path]
            if spec_paths is not None
            else _default_op_scoped_kind(prefix, name)
        )
        namespace = {"shape": shape_ns, "dtype": dtype_ns, "attrs": attrs_ns}[prefix]
        fact = namespace[name]
        return Fact(fact.path, kind)

    if path in GLOBAL_FACT_PATHS:
        return _global_fact(path)
    if prefix == "lib" and name:
        return Fact(path, FactKind.BOOL)
    if prefix == "quant" and path.startswith("quant.fields."):
        return Fact(path, FactKind.ANY)

    raise PolicyError(f"unknown fact {path!r}{_suggest(path, GLOBAL_FACT_PATHS)}")


def _default_op_scoped_kind(prefix: str, name: str) -> FactKind:
    if prefix == "shape":
        return FactKind.INT
    if prefix == "dtype":
        return FactKind.DTYPE
    return FactKind.ANY


def _global_fact(path: str) -> Fact:
    from phyai.kernel import facts as facts_module

    prefix, _, name = path.partition(".")
    if not name:
        return getattr(facts_module, path)
    namespace = getattr(facts_module, prefix)
    return getattr(namespace, name)


def _suggest(path: str, known: Iterable[str]) -> str:
    options = sorted(known)
    hint = suggest_name(path, options)
    return hint if hint else f" (known: {options})"


def compile_matcher(
    matcher: Mapping[str, Any],
    *,
    spec_paths: Mapping[str, FactKind] | None = None,
    prefix: str = "",
) -> Predicate:
    """Compile one policy matcher into a predicate."""

    if not isinstance(matcher, Mapping):
        raise PolicyError(f"a matcher must be a mapping, got {type(matcher).__name__}")

    parts: list[Predicate] = []
    for raw_key, value in matcher.items():
        key = str(raw_key)

        if key in _COMBINATORS:
            if prefix:
                raise PolicyError(
                    f"{key!r} is a boolean combinator and cannot appear inside "
                    f"the {prefix!r} namespace"
                )
            parts.append(_compile_combinator(key, value, spec_paths=spec_paths))
            continue

        path = f"{prefix}.{key}" if prefix else key

        # Flatten nested namespaces into dotted fact paths.
        if isinstance(value, Mapping):
            if not prefix and key not in _NAMESPACES:
                raise PolicyError(
                    f"{key!r} is not a namespace, so it cannot take a mapping "
                    f"value; known namespaces: {sorted(_NAMESPACES)}"
                )
            parts.append(compile_matcher(value, spec_paths=spec_paths, prefix=path))
            continue

        fact = _fact_for(path, spec_paths)
        parts.append(_literal_predicate(fact, value, path))

    if not parts:
        return TRUE
    return parts[0] if len(parts) == 1 else all_of(*parts)


def _literal_predicate(fact: Fact, value: Any, path: str) -> Predicate:
    try:
        return predicate_from_literal(fact, value)
    except (TypeError, ValueError) as exc:
        raise PolicyError(
            f"cannot match {path!r} ({fact.kind.value}) against {value!r}: {exc}"
        ) from exc


def _compile_combinator(
    key: str, value: Any, *, spec_paths: Mapping[str, FactKind] | None
) -> Predicate:
    if key == "when":
        if not isinstance(value, Mapping) or set(value) != {"if", "then"}:
            raise PolicyError("`when` takes exactly {if: <matcher>, then: <matcher>}")
        from phyai.kernel.predicate import implies

        return implies(
            compile_matcher(value["if"], spec_paths=spec_paths),
            compile_matcher(value["then"], spec_paths=spec_paths),
        )

    if not isinstance(value, (list, tuple)):
        raise PolicyError(
            f"{key!r} takes a list of matchers, got {type(value).__name__}"
        )
    clauses = [compile_matcher(item, spec_paths=spec_paths) for item in value]
    if not clauses:
        raise PolicyError(f"{key!r} needs at least one clause")
    if key == "all_of":
        return all_of(*clauses)
    if key == "any_of":
        return any_of(*clauses)
    return none_of(*clauses)


@dataclass(frozen=True)
class Rule:
    """Describe a conditional kernel preference or override."""

    rule_id: str
    priority: int
    when: Predicate
    prefer: tuple[str, ...] = ()
    restrict_to: str | None = None
    use: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    strict: bool = False
    source_match: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", str(self.rule_id))
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(
            self, "prefer", tuple(str(item).lower() for item in self.prefer)
        )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "strict", bool(self.strict))
        if self.use is not None:
            object.__setattr__(self, "use", str(self.use).lower())
        if self.restrict_to is not None:
            object.__setattr__(self, "restrict_to", str(self.restrict_to).lower())

    def matches(self, facts) -> bool:
        return self.when.eval(facts) is None


@dataclass(frozen=True)
class Decision:
    """Store the policy decision for one call."""

    #: Ordered kernel IDs to try.
    candidates: tuple[str, ...] = ()
    #: ``True`` for overrides: an unusable choice is an error, not a fallback.
    strict: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)
    matched_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidates", tuple(str(item).lower() for item in self.candidates)
        )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(
            self, "matched_rules", tuple(str(item) for item in self.matched_rules)
        )


@dataclass(frozen=True)
class Policy:
    """Store compiled kernel rules and optional load-time quantization."""

    profile: str = "static"
    fallback: str = "reference"
    rules: tuple[Rule, ...] = ()
    overrides: tuple[Rule, ...] = ()
    source: str | None = None
    quant_plan: QuantPlan | None = None

    #: Cached policy fingerprint.
    version: str = field(default="", compare=False, repr=False)

    def __post_init__(self) -> None:
        profile = str(self.profile).lower()
        fallback = str(self.fallback).lower()
        if profile not in {"static", "autotune"}:
            raise PolicyError("profile must be 'static' or 'autotune'")
        if fallback not in {"reference", "error"}:
            raise PolicyError("defaults.fallback must be 'reference' or 'error'")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "rules", tuple(self.rules or ()))
        object.__setattr__(self, "overrides", tuple(self.overrides or ()))
        if self.quant_plan is not None:
            from phyai.layers.quant.plan import QuantPlan

            if not isinstance(self.quant_plan, QuantPlan):
                raise PolicyError("quant_plan must be a QuantPlan or None")
        object.__setattr__(self, "version", self.compute_version())

    def decide(self, facts, catalog: Catalog) -> Decision:
        """Return the highest-priority matching override or rule decision."""

        override = _highest(self.overrides, facts, kind="override")
        if override is not None:
            candidates = _expand(override, catalog)
            return Decision(candidates, True, override.params, (override.rule_id,))

        rule = _highest(self.rules, facts, kind="rule")
        if rule is None:
            return Decision()
        return Decision(_expand(rule, catalog), False, rule.params, (rule.rule_id,))

    def facts_used(self) -> frozenset[str]:
        """Return every fact path read by a rule or override."""

        paths: frozenset[str] = frozenset()
        for rule in (*self.rules, *self.overrides):
            paths |= rule.when.facts_used()
        return paths

    def compute_version(self) -> str:
        """Return the fingerprint used by selection and autotune caches."""

        payload = {
            "profile": self.profile,
            "fallback": self.fallback,
            "rules": [_rule_payload(item) for item in self.rules],
            "overrides": [_rule_payload(item) for item in self.overrides],
        }
        if self.quant_plan is not None:
            from phyai.layers.quant.policy import quant_plan_payload

            payload["quantization"] = quant_plan_payload(self.quant_plan)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def _rule_payload(rule: Rule) -> dict[str, object]:
    return {
        "id": rule.rule_id,
        "priority": rule.priority,
        "when": rule.when.render(),
        "prefer": list(rule.prefer),
        "restrict_to": rule.restrict_to,
        "use": rule.use,
        "params": dict(rule.params),
    }


def _highest(rules: Sequence[Rule], facts, *, kind: str) -> Rule | None:
    matching = [rule for rule in rules if rule.matches(facts)]
    if not matching:
        return None
    top = max(rule.priority for rule in matching)
    winners = [rule for rule in matching if rule.priority == top]
    if len(winners) > 1:
        ids = sorted(rule.rule_id for rule in winners)
        raise PolicyError(
            f"conflicting {kind}s at priority {top} for this call: {ids}. "
            f"Give one of them a different priority."
        )
    return winners[0]


def _expand(rule: Rule, catalog: Catalog) -> tuple[str, ...]:
    if rule.use is not None:
        return (rule.use,)
    if rule.restrict_to is not None:
        return catalog.match_ids(rule.restrict_to)
    return rule.prefer


def policy_from_mapping(
    value: Mapping[str, Any],
    catalog: Catalog,
    *,
    source: str | None = None,
) -> Policy:
    """Compile and fully validate a policy mapping."""

    if not isinstance(value, Mapping):
        raise PolicyError("a policy document must be a mapping")

    unknown = set(value) - {
        "schema",
        "profile",
        "defaults",
        "rules",
        "overrides",
        "quantization",
    }
    if unknown:
        raise PolicyError(f"unknown top-level field(s): {sorted(unknown)}")

    schema = value.get("schema", SCHEMA)
    if schema not in _SCHEMA_VALUES:
        raise PolicyError(
            f"unsupported schema {schema!r}; this build understands {SCHEMA!r}"
        )

    defaults = value.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise PolicyError("defaults must be a mapping")
    unknown_defaults = set(defaults) - {"fallback"}
    if unknown_defaults:
        raise PolicyError(f"unknown defaults field(s): {sorted(unknown_defaults)}")

    rules = tuple(
        _compile_rule(raw, index, catalog, strict=False)
        for index, raw in enumerate(value.get("rules") or ())
    )
    overrides = tuple(
        _compile_rule(raw, index, catalog, strict=True)
        for index, raw in enumerate(value.get("overrides") or ())
    )
    quant_plan = None
    if value.get("quantization") is not None:
        from phyai.layers.quant.policy import quant_plan_from_mapping

        try:
            quant_plan = quant_plan_from_mapping(value["quantization"])
        except ValueError as error:
            raise PolicyError(str(error)) from error

    return Policy(
        profile=str(value.get("profile", "static")),
        fallback=str(defaults.get("fallback", "reference")),
        rules=rules,
        overrides=overrides,
        source=source,
        quant_plan=quant_plan,
    )


def _compile_rule(raw: Any, index: int, catalog: Catalog, *, strict: bool) -> Rule:
    kind = "override" if strict else "rule"
    if not isinstance(raw, Mapping):
        raise PolicyError(f"{kind} {index} must be a mapping")

    allowed = _OVERRIDE_FIELDS if strict else _RULE_FIELDS
    unknown = set(raw) - allowed
    if unknown:
        raise PolicyError(
            f"{kind} {raw.get('id', index)!r}: unknown field(s) {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )

    rule_id = str(raw.get("id", f"{kind}-{index}"))
    matcher = raw.get("match") or {}
    if not isinstance(matcher, Mapping):
        raise PolicyError(f"{kind} {rule_id!r}: match must be a mapping")

    # An explicit operation enables validation of operation-scoped facts.
    op_name = matcher.get("op")
    spec_paths: dict[str, FactKind] | None = None
    if isinstance(op_name, str):
        spec = catalog.maybe_op(op_name)
        if spec is None:
            raise PolicyError(
                f"{kind} {rule_id!r}: unknown op {op_name!r} "
                f"(known: {[item.name for item in catalog.ops()]})"
            )
        spec_paths = {path: spec.kind_of(path) for path in spec.known_paths()}

    try:
        when = compile_matcher(matcher, spec_paths=spec_paths)
    except PolicyError as exc:
        raise PolicyError(f"{kind} {rule_id!r}: {exc}") from None

    prefer = raw.get("prefer") or ()
    if isinstance(prefer, str):
        prefer = (prefer,)
    use = raw.get("use")
    restrict_to = raw.get("restrict_to")

    if strict:
        if bool(use) == bool(restrict_to):
            raise PolicyError(
                f"override {rule_id!r} needs exactly one of `use` "
                f"(a single kernel id) or `restrict_to` (a kernel-id glob)"
            )
    elif bool(prefer) == bool(restrict_to):
        # Soft rules choose either ordered preferences or a kernel glob.
        raise PolicyError(
            f"rule {rule_id!r} needs exactly one of `prefer` (an ordered "
            f"kernel-id list) or `restrict_to` (a kernel-id glob)"
        )

    for kernel_id in (*prefer, *((use,) if use else ())):
        if catalog.maybe_get(kernel_id) is None:
            raise PolicyError(
                f"{kind} {rule_id!r} references unknown kernel {kernel_id!r}"
                f"{_suggest(str(kernel_id), catalog.kernel_ids())}"
            )
    if restrict_to and not catalog.match_ids(restrict_to):
        raise PolicyError(
            f"{kind} {rule_id!r}: restrict_to {restrict_to!r} matches no kernel "
            f"(available: {list(catalog.kernel_ids())})"
        )

    return Rule(
        rule_id=rule_id,
        priority=int(raw.get("priority", 0)),
        when=when,
        prefer=tuple(prefer),
        restrict_to=restrict_to,
        use=use,
        params=dict(raw.get("params") or {}),
        strict=strict,
        source_match=dict(matcher),
    )


def load_policy(path: str | Path | None, catalog: Catalog) -> Policy:
    """Load a policy file, or return the empty deterministic default."""

    if path is None:
        return Policy()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("PyYAML is required to load a kernel policy") from exc

    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    try:
        return policy_from_mapping(loaded, catalog, source=str(resolved))
    except PolicyError as exc:
        raise PolicyError(f"{resolved}: {exc}") from None


__all__ = [
    "Decision",
    "Policy",
    "PolicyError",
    "Rule",
    "SCHEMA",
    "compile_matcher",
    "load_policy",
    "policy_from_mapping",
]
