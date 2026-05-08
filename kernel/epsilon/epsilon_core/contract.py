"""Python ↔ Rust ManifoldPayload contract.

This module is the Python-side counterpart to the Rust struct
``ManifoldPayload<D>`` defined in
``kernel/epsilon/epsilon/crates/epsilon/src/manifold.rs:264``.

The wire format is documented in
``schemas/manifold_payload.schema.json`` and is the single source of truth for
cross-language serialization. The Python reference implementation must produce
and accept payloads that round-trip through this schema.

Use :func:`encode` to serialize a payload to a dict suitable for ``json.dumps``,
and :func:`decode` to validate and parse one. :func:`validate` performs schema
validation only and returns the input unchanged on success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


SCHEMA_ID = "https://epsilon-hollow/schemas/manifold_payload.schema.json"


@dataclass
class ManifoldPayload:
    """Mirrors Rust's ``ManifoldPayload<D>``.

    Field order and semantics must stay byte-compatible with the Rust struct.
    See the JSON schema for invariants (Euler characteristic, point_count
    bounds, etc.).
    """

    dim: int
    point_count: int
    points: list[list[float]]
    signature_b0: int = 0
    signature_b1: int = 0
    signature_b2: int = 0
    liveness_anchor: float = 1.0

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        *,
        dim: int,
        b0: int = 0,
        b1: int = 0,
        b2: int = 1,
        liveness_anchor: float = 1.0,
    ) -> "ManifoldPayload":
        pts = [list(p) for p in points]
        for p in pts:
            if len(p) != dim:
                raise ValueError(
                    f"Point dimension mismatch: got {len(p)}, expected {dim}"
                )
        return cls(
            dim=dim,
            point_count=len(pts),
            points=pts,
            signature_b0=b0,
            signature_b1=b1,
            signature_b2=b2,
            liveness_anchor=liveness_anchor,
        )


def encode(payload: ManifoldPayload) -> dict[str, Any]:
    """Serialize to the wire shape defined by ``manifold_payload.schema.json``."""
    return {
        "dim": payload.dim,
        "point_count": payload.point_count,
        "points": payload.points,
        "signature_b0": payload.signature_b0,
        "signature_b1": payload.signature_b1,
        "signature_b2": payload.signature_b2,
        "liveness_anchor": payload.liveness_anchor,
    }


def validate(obj: dict[str, Any]) -> dict[str, Any]:
    """Schema validation. Raises ``ValueError`` on any contract violation.

    This is a hand-rolled checker so the contract test stays dependency-free.
    For richer validation use ``jsonschema`` against the JSON schema directly.
    """
    required = {
        "dim",
        "point_count",
        "points",
        "signature_b0",
        "signature_b1",
        "signature_b2",
        "liveness_anchor",
    }
    missing = required - set(obj)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    extra = set(obj) - required
    if extra:
        raise ValueError(f"unexpected fields: {sorted(extra)}")

    if not isinstance(obj["dim"], int) or obj["dim"] < 1:
        raise ValueError("dim must be a positive integer")
    if not isinstance(obj["point_count"], int) or obj["point_count"] < 0:
        raise ValueError("point_count must be a non-negative integer")
    if not isinstance(obj["points"], list):
        raise ValueError("points must be a list")

    dim = obj["dim"]
    for i, p in enumerate(obj["points"]):
        if not isinstance(p, list) or len(p) != dim:
            raise ValueError(f"points[{i}] is not a length-{dim} list")
        for v in p:
            if not isinstance(v, (int, float)):
                raise ValueError(f"points[{i}] contains non-numeric value")

    if obj["point_count"] > len(obj["points"]):
        raise ValueError("point_count exceeds len(points)")

    for k in ("signature_b0", "signature_b1", "signature_b2"):
        if not isinstance(obj[k], int) or obj[k] < 0:
            raise ValueError(f"{k} must be a non-negative integer")

    la = obj["liveness_anchor"]
    if not isinstance(la, (int, float)) or la < 0:
        raise ValueError("liveness_anchor must be a non-negative number")

    return obj


def decode(obj: dict[str, Any]) -> ManifoldPayload:
    """Validate then construct a :class:`ManifoldPayload`."""
    validate(obj)
    return ManifoldPayload(
        dim=obj["dim"],
        point_count=obj["point_count"],
        points=[list(p) for p in obj["points"]],
        signature_b0=obj["signature_b0"],
        signature_b1=obj["signature_b1"],
        signature_b2=obj["signature_b2"],
        liveness_anchor=float(obj["liveness_anchor"]),
    )


__all__ = ["ManifoldPayload", "encode", "decode", "validate", "SCHEMA_ID"]
