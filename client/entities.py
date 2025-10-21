"""Client side entity representations mirroring the server state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Segment:
    """Represents a single body segment for rendering."""

    x: float
    y: float


@dataclass
class SnakeEntity:
    """Renderable snake state synchronised from the server."""

    id: int
    name: str
    color: Tuple[int, int, int]
    length: float
    segments: List[Segment] = field(default_factory=list)
    alive: bool = True

    def update_from_snapshot(self, snapshot: dict) -> None:
        self.name = snapshot.get("name", self.name)
        self.color = tuple(snapshot.get("color", self.color))  # type: ignore[arg-type]
        self.length = snapshot.get("length", self.length)
        self.alive = snapshot.get("alive", self.alive)
        self.segments = [
            Segment(float(segment["x"]), float(segment["y"]))
            for segment in snapshot.get("segments", [])
        ]


@dataclass
class PelletEntity:
    """Renderable pellet state."""

    id: int
    x: float
    y: float
    value: int


class EntityStore:
    """Maintain active snakes and pellets for the renderer."""

    def __init__(self) -> None:
        self.snakes: Dict[int, SnakeEntity] = {}
        self.pellets: Dict[int, PelletEntity] = {}

    def update_from_snapshot(self, snapshot: dict) -> None:
        self._update_snakes(snapshot.get("snakes", []))
        self._update_pellets(snapshot.get("pellets", []))

    def _update_snakes(self, snakes_payload: List[dict]) -> None:
        existing_ids = set(self.snakes.keys())
        visible_ids = set()
        for payload in snakes_payload:
            snake_id = int(payload["id"])
            visible_ids.add(snake_id)
            entity = self.snakes.get(snake_id)
            if entity is None:
                entity = SnakeEntity(
                    id=snake_id,
                    name=payload.get("name", "Unknown"),
                    color=tuple(payload.get("color", (255, 255, 255))),
                    length=float(payload.get("length", 0.0)),
                )
                self.snakes[snake_id] = entity
            entity.update_from_snapshot(payload)
        for snake_id in existing_ids - visible_ids:
            self.snakes.pop(snake_id, None)

    def _update_pellets(self, pellets_payload: List[dict]) -> None:
        existing_ids = set(self.pellets.keys())
        visible_ids = set()
        for payload in pellets_payload:
            pellet_id = int(payload["id"])
            visible_ids.add(pellet_id)
            entity = self.pellets.get(pellet_id)
            if entity is None:
                entity = PelletEntity(
                    id=pellet_id,
                    x=float(payload.get("x", 0.0)),
                    y=float(payload.get("y", 0.0)),
                    value=int(payload.get("value", 1)),
                )
                self.pellets[pellet_id] = entity
            else:
                entity.x = float(payload.get("x", entity.x))
                entity.y = float(payload.get("y", entity.y))
                entity.value = int(payload.get("value", entity.value))
        for pellet_id in existing_ids - visible_ids:
            self.pellets.pop(pellet_id, None)
