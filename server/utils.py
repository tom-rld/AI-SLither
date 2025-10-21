"""Utility primitives used by the authoritative game server."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence

from . import constants


@dataclass
class Vec2:
    """A light‑weight two dimensional vector used for geometry operations.

    The class implements the operations that are required to simulate the
    serpents: addition, subtraction, scaling, normalisation and distance
    computations. The implementation intentionally keeps the surface area
    minimal to avoid unnecessary overhead inside the hot update loop.
    """

    x: float
    y: float

    def copy(self) -> "Vec2":
        """Return a shallow copy of the vector."""

        return Vec2(self.x, self.y)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def length(self) -> float:
        """Return the Euclidean length of the vector."""

        return math.hypot(self.x, self.y)

    def distance_to(self, other: "Vec2") -> float:
        """Return the distance between this vector and ``other``."""

        return (self - other).length()

    def normalized(self) -> "Vec2":
        """Return a normalised copy of the vector.

        The function safely handles the zero vector by returning ``Vec2(0, 0)``.
        """

        length = self.length()
        if length == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / length, self.y / length)

    def to_tuple(self) -> tuple[float, float]:
        """Return the vector as an ``(x, y)`` tuple."""

        return self.x, self.y


def random_point_in_world() -> Vec2:
    """Return a uniformly random point inside the world circle."""

    radius = constants.WORLD_RADIUS * math.sqrt(random.random())
    angle = random.random() * 2 * math.pi
    return Vec2(math.cos(angle) * radius, math.sin(angle) * radius)


def clamp_world_position(position: Vec2) -> Vec2:
    """Clamp ``position`` so that it remains inside the world circle."""

    if position.length() <= constants.WORLD_RADIUS:
        return position
    return position.normalized() * constants.WORLD_RADIUS


def lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    """Linearly interpolate between ``a`` and ``b`` by the factor ``t``."""

    return Vec2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


def polyline_length(points: Sequence[Vec2]) -> float:
    """Return the total length of a polyline defined by ``points``."""

    return sum((points[i] - points[i - 1]).length() for i in range(1, len(points)))


def subdivide_polyline(points: Sequence[Vec2], segment_length: float) -> List[Vec2]:
    """Return a new polyline resampled to a fixed ``segment_length``.

    The helper is used to derive the snake body segments from the current head
    trajectory. The algorithm walks along the polyline and emits points whenever
    enough distance has been accumulated.
    """

    if not points:
        return []

    result: List[Vec2] = [points[0].copy()]
    accumulated = 0.0
    for start, end in zip(points, points[1:]):
        segment = end - start
        seg_length = segment.length()
        if seg_length == 0:
            continue
        direction = segment * (1.0 / seg_length)
        distance_remaining = seg_length
        while accumulated + distance_remaining >= segment_length:
            overshoot = segment_length - accumulated
            start = start + direction * overshoot
            distance_remaining -= overshoot
            result.append(start.copy())
            accumulated = 0.0
        accumulated += distance_remaining
    return result


def iter_pairwise(iterable: Iterable[Vec2]):
    """Yield pairwise elements from ``iterable`` as ``(current, next)`` tuples."""

    iterator = iter(iterable)
    previous = next(iterator, None)
    for item in iterator:
        if previous is not None:
            yield previous, item
        previous = item
