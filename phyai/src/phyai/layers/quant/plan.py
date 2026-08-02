"""Layer->scheme mapping: an ordered rule table."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Literal

from phyai.layers.quant.scheme import QuantScheme

MatchKind = Literal["name", "glob", "regex", "module_cls"]


@dataclass(frozen=True)
class Matcher:
    kind: MatchKind
    pattern: str

    def matches(self, prefix: str, module_cls: type | None) -> bool:
        if self.kind == "name":
            return self.pattern == prefix or prefix.split(".")[-1] == self.pattern
        if self.kind == "glob":
            return fnmatch.fnmatch(prefix, self.pattern)
        if self.kind == "regex":
            return re.match(self.pattern, prefix) is not None
        if self.kind == "module_cls":
            return module_cls is not None and self.pattern in module_cls.__name__
        raise ValueError(f"unknown Matcher kind {self.kind!r}")


@dataclass(frozen=True)
class Rule:
    matcher: Matcher
    scheme: QuantScheme | None  # None = SKIP


@dataclass(frozen=True)
class QuantPlan:
    """Ordered rule table; :meth:`resolve` returns the first matching scheme.

    Build one by pairing a :class:`Matcher` with a :class:`QuantScheme`
    (a ``None`` scheme means SKIP -> the layer stays bf16)::

        from phyai.layers.quant.granularity import Granularity
        from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant

        fp8 = QuantScheme(
            weight=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL),
            input=TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL, dynamic=True),
        )
        plan = QuantPlan(
            rules=(
                Rule(Matcher("name", "lm_head"), None),  # skip -> bf16
                Rule(Matcher("glob", "*.mlp.*"), fp8),  # mlp linears -> fp8
            ),
            default=None,  # unmatched layers -> bf16
        )
    """

    rules: tuple[Rule, ...]
    default: QuantScheme | None

    def resolve(
        self, prefix: str, module_cls: type | None = None
    ) -> QuantScheme | None:
        for rule in self.rules:
            if rule.matcher.matches(prefix, module_cls):
                return rule.scheme
        return self.default


__all__ = ["MatchKind", "Matcher", "Rule", "QuantPlan"]
