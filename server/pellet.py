"""Pellet entity definition."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

from . import constants, utils

_id_counter = itertools.count(1)


@dataclass
class Pellet:
    """A food pellet that snakes can consume to grow."""

    id: int
    position: utils.Vec2
    value: int

    @classmethod
    def spawn_random(cls) -> "Pellet":
        """Create a pellet at a random position with a random value."""

        return cls(
            id=next(_id_counter),
            position=utils.random_point_in_world(),
            value=random.randint(constants.PELLET_MIN_VALUE, constants.PELLET_MAX_VALUE),
        )

    @classmethod
    def from_position(cls, position: utils.Vec2, value: int) -> "Pellet":
        """Create a pellet at a specific ``position``."""

        return cls(id=next(_id_counter), position=position.copy(), value=value)

    def to_dict(self) -> dict[str, float | int]:
        """Serialise the pellet to a JSON friendly dictionary."""

        return {"id": self.id, "x": self.position.x, "y": self.position.y, "value": self.value}
