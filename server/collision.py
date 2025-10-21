"""Collision helpers for the game server."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .snake import Snake
from . import constants


def _circle_intersection(ax: float, ay: float, ar: float, bx: float, by: float, br: float) -> bool:
    """Return ``True`` if two circles intersect."""

    dx = ax - bx
    dy = ay - by
    distance_sq = dx * dx + dy * dy
    radius_sum = ar + br
    return distance_sq <= radius_sum * radius_sum


def detect_head_collisions(snakes: Iterable[Snake]) -> List[Tuple[Snake, Snake]]:
    """Return all ``(attacker, victim)`` pairs whose head hit another body."""

    snakes = [snake for snake in snakes if snake.alive]
    collisions: List[Tuple[Snake, Snake]] = []
    for attacker in snakes:
        head, radius = attacker.head_circle()
        for victim in snakes:
            if attacker is victim:
                # Skip the first few segments to avoid accidental head overlap.
                segments = victim.body[3:]
            else:
                segments = victim.body
            for segment in segments:
                if _circle_intersection(head.x, head.y, radius, segment.x, segment.y, victim.radius):
                    collisions.append((attacker, victim))
                    break
            else:
                continue
            break
    return collisions


def clamp_inside_world(snakes: Iterable[Snake]) -> None:
    """Ensure that all snakes remain inside the world bounds."""

    for snake in snakes:
        if not snake.alive:
            continue
        head, _ = snake.head_circle()
        if head.length() > constants.WORLD_RADIUS:
            snake.head_position = snake.head_position.normalized() * constants.WORLD_RADIUS
            snake.body[0] = snake.head_position.copy()
