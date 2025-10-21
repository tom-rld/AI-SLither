"""Helpers to smooth server snapshots before rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .entities import SnakeEntity, Segment


@dataclass
class InterpolatedSnake:
    """A lightweight representation used purely for drawing."""

    id: int
    color: tuple[int, int, int]
    segments: List[Segment]
    name: str
    length: float


class Interpolator:
    """Currently a passthrough interpolation implementation.

    The client renders the most recent snapshot verbatim. The class is designed
    to be easily extended with proper time based interpolation if required in
    the future.
    """

    def build(self, snakes: Dict[int, SnakeEntity]) -> List[InterpolatedSnake]:
        interpolated: List[InterpolatedSnake] = []
        for snake in snakes.values():
            interpolated.append(
                InterpolatedSnake(
                    id=snake.id,
                    color=snake.color,
                    segments=list(snake.segments),
                    name=snake.name,
                    length=snake.length,
                )
            )
        return interpolated
