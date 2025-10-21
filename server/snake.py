"""Snake entity implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import random
from typing import List

from . import constants, utils

_id_counter = itertools.count(1)


@dataclass
class SnakeInput:
    """Represents the latest player input processed by the server."""

    angle: float = 0.0
    boost: bool = False


@dataclass
class Snake:
    """Authoritative representation of a snake controlled by a player."""

    name: str
    color: tuple[int, int, int]
    head_position: utils.Vec2
    angle: float = 0.0
    length: float = 40.0
    alive: bool = True
    input: SnakeInput = field(default_factory=SnakeInput)
    score: float = 0.0
    id: int = field(default_factory=lambda: next(_id_counter))

    def __post_init__(self) -> None:
        self.segment_spacing = constants.SNAKE_RADIUS * constants.SEGMENT_SPACING_FACTOR
        self.body: List[utils.Vec2] = [self.head_position.copy()]
        self._mass_buffer = 0.0
        self._pending_growth = 0.0
        self.score = self.length
        self._ensure_body_segments()

    @property
    def radius(self) -> float:
        """Collision radius of each segment."""

        return constants.SNAKE_RADIUS

    @property
    def direction(self) -> utils.Vec2:
        """Return the current heading direction."""

        return utils.Vec2(math.cos(self.angle), math.sin(self.angle))

    def set_input(self, angle: float, boost: bool) -> None:
        """Update the desired heading and boost flag."""

        self.input.angle = angle
        self.input.boost = boost

    def add_length(self, amount: float) -> None:
        """Increase the desired snake length by ``amount``."""

        self._pending_growth += amount

    def kill(self) -> None:
        """Mark the snake as dead."""

        self.alive = False

    def respawn(self, position: utils.Vec2) -> None:
        """Respawn the snake at ``position`` with default properties."""

        self.head_position = position.copy()
        self.angle = random.random() * 2 * math.pi
        self.length = 40.0
        self.alive = True
        self._pending_growth = 0.0
        self._mass_buffer = 0.0
        self.body = [self.head_position.copy()]
        self._ensure_body_segments()

    def _ensure_body_segments(self) -> None:
        """Ensure that ``self.body`` matches the desired length."""

        target_count = max(2, int(self.length / self.segment_spacing))
        current_count = len(self.body)
        if current_count < target_count:
            tail = self.body[-1]
            for _ in range(target_count - current_count):
                self.body.append(tail.copy())
        elif current_count > target_count:
            self.body = self.body[:target_count]

    def _trim_to_length(self) -> None:
        """Trim or extend the body so its total length matches ``self.length``."""

        target_count = max(2, int(self.length / self.segment_spacing))
        if len(self.body) != target_count:
            self._ensure_body_segments()
        # Adjust tail to better match precise length by cutting the last segment
        total_length = (len(self.body) - 1) * self.segment_spacing
        if total_length > 0:
            overshoot = total_length - self.length
            if overshoot > 0 and len(self.body) >= 2:
                direction = self.body[-2] - self.body[-1]
                distance = direction.length()
                if distance > 0:
                    self.body[-1] = self.body[-2] - direction.normalized() * (
                        self.segment_spacing - overshoot
                    )

    def tick(self) -> list[utils.Vec2]:
        """Advance the snake simulation by one tick.

        Returns a list of positions where boost pellets should spawn.
        """

        if not self.alive:
            return []

        self.angle = self.input.angle
        direction = self.direction
        speed = constants.BASE_SPEED

        boosting = self.input.boost and self.length > 10.0
        if boosting:
            speed *= constants.BOOST_MULTIPLIER

        displacement = direction * speed
        self.head_position = utils.clamp_world_position(self.head_position + displacement)
        self.body[0] = self.head_position.copy()

        for index in range(1, len(self.body)):
            prev = self.body[index - 1]
            curr = self.body[index]
            direction_to_prev = prev - curr
            distance = direction_to_prev.length()
            if distance == 0:
                self.body[index] = prev.copy()
                continue
            self.body[index] = prev - direction_to_prev * (self.segment_spacing / distance)

        self.length += self._pending_growth
        self._pending_growth = 0.0

        pellets_to_spawn: list[utils.Vec2] = []
        if boosting:
            if self.length > constants.BOOST_COST_PER_TICK * 1.5:
                self.length -= constants.BOOST_COST_PER_TICK
                self._mass_buffer += constants.BOOST_COST_PER_TICK
                tail_position = self.body[-1].copy()
                while self._mass_buffer >= 1.0:
                    pellets_to_spawn.append(tail_position.copy())
                    self._mass_buffer -= 1.0
        else:
            self._mass_buffer = min(self._mass_buffer, 1.0)

        self._trim_to_length()
        self.score = max(self.score, self.length)
        return pellets_to_spawn

    def to_snapshot(self) -> dict:
        """Return a snapshot representation for clients."""

        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "length": self.length,
            "alive": self.alive,
            "segments": [
                {"x": point.x, "y": point.y}
                for point in self.body
            ],
        }

    def head_circle(self) -> tuple[utils.Vec2, float]:
        """Return the head circle used for collision detection."""

        return self.head_position, self.radius

    def body_circles(self) -> List[tuple[utils.Vec2, float]]:
        """Return all body segments as circles for collision tests."""

        return [(point, self.radius) for point in self.body]
