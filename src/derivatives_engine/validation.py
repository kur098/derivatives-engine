"""Input validation helpers.

Every public entry point funnels user input through these functions so that
invalid values fail early with a descriptive :class:`ValidationError`
(spec FR-01, NFR-06, AT-07).
"""

from __future__ import annotations

import math
import numbers
from enum import Enum
from typing import Mapping, TypeVar

E = TypeVar("E", bound=Enum)


class ValidationError(ValueError):
    """Raised when inputs or configurations are invalid."""


class NumericalError(RuntimeError):
    """Raised when a numerical method cannot produce a valid result."""


def require_finite(value: object, name: str) -> float:
    """Return ``value`` as a float, rejecting bools, non-numbers, NaN and inf."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(
            f"{name} must be a real number, got {type(value).__name__} ({value!r})."
        )
    fval = float(value)
    if math.isnan(fval):
        raise ValidationError(f"{name} must not be NaN.")
    if math.isinf(fval):
        raise ValidationError(f"{name} must be finite, got {fval}.")
    return fval


def require_positive(value: float, name: str) -> float:
    if not value > 0.0:
        raise ValidationError(f"{name} must be > 0, got {value}.")
    return value


def require_non_negative(value: float, name: str) -> float:
    if not value >= 0.0:
        raise ValidationError(f"{name} must be >= 0, got {value}.")
    return value


def require_int(value: object, name: str, minimum: int | None = None) -> int:
    """Validate an integer (numpy integers allowed, bools and floats rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})."
        )
    ival = int(value)
    if minimum is not None and ival < minimum:
        raise ValidationError(f"{name} must be >= {minimum}, got {ival}.")
    return ival


def require_seed(value: object) -> int | None:
    if value is None:
        return None
    return require_int(value, "seed", minimum=0)


def parse_enum(
    value: object,
    enum_cls: type[E],
    name: str,
    aliases: Mapping[str, str] | None = None,
) -> E:
    """Parse ``value`` into ``enum_cls`` (case-insensitive, optional aliases)."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if aliases and key in aliases:
            key = aliases[key]
        try:
            return enum_cls(key)
        except ValueError:
            pass
    allowed = ", ".join(repr(m.value) for m in enum_cls)  # type: ignore[attr-defined]
    raise ValidationError(f"Unsupported {name} {value!r}; expected one of {allowed}.")
