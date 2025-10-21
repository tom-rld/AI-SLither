"""Authoritative game world simulation."""

from __future__ import annotations

import asyncio
import random
from typing import Dict, Iterable, List, Optional

from . import collision, constants, protocol, utils
from .pellet import Pellet
from .snake import Snake


class World:
    """Holds all entities and advances the simulation on every tick."""

    def __init__(self) -> None:
        self.tick: int = 0
        self.snakes: Dict[int, Snake] = {}
        self.pellets: Dict[int, Pellet] = {}
        self._respawn_counters: Dict[int, int] = {}
        self._populate_initial_pellets()

    def _populate_initial_pellets(self) -> None:
        for _ in range(constants.INITIAL_PELLET_COUNT):
            pellet = Pellet.spawn_random()
            self.pellets[pellet.id] = pellet

    def _maintain_pellet_population(self) -> None:
        while len(self.pellets) < constants.INITIAL_PELLET_COUNT:
            pellet = Pellet.spawn_random()
            self.pellets[pellet.id] = pellet

    def add_snake(self, name: str) -> Snake:
        color = tuple(random.randint(64, 240) for _ in range(3))
        snake = Snake(name=name, color=color, head_position=utils.random_point_in_world())
        snake.angle = random.random() * 2 * 3.14159265
        snake.set_input(snake.angle, False)
        self.snakes[snake.id] = snake
        return snake

    def remove_snake(self, snake_id: int) -> None:
        self.snakes.pop(snake_id, None)
        self._respawn_counters.pop(snake_id, None)

    def set_player_input(self, snake_id: int, angle: float, boost: bool) -> None:
        snake = self.snakes.get(snake_id)
        if snake and snake.alive:
            snake.set_input(angle, boost)

    def _handle_pellet_collisions(self, snake: Snake) -> None:
        consumed: List[int] = []
        head, radius = snake.head_circle()
        for pellet in self.pellets.values():
            if head.distance_to(pellet.position) <= radius * 1.3:
                consumed.append(pellet.id)
                snake.add_length(pellet.value)
        for pellet_id in consumed:
            self.pellets.pop(pellet_id, None)

    def _drop_snake_pellets(self, snake: Snake) -> None:
        for segment in snake.body:
            pellet = Pellet.from_position(segment, max(1, int(snake.length / len(snake.body))))
            self.pellets[pellet.id] = pellet

    def update(self) -> None:
        self.tick += 1
        for snake in list(self.snakes.values()):
            if snake.alive:
                boost_pellets = snake.tick()
                for position in boost_pellets:
                    pellet = Pellet.from_position(position, int(constants.BOOST_PELLET_VALUE))
                    self.pellets[pellet.id] = pellet
                self._handle_pellet_collisions(snake)
            else:
                counter = self._respawn_counters.get(snake.id, constants.RESPAWN_DELAY_TICKS)
                counter -= 1
                if counter <= 0:
                    snake.respawn(utils.random_point_in_world())
                    self._respawn_counters.pop(snake.id, None)
                else:
                    self._respawn_counters[snake.id] = counter

        collisions = collision.detect_head_collisions(self.snakes.values())
        for attacker, victim in collisions:
            if attacker.alive:
                attacker.kill()
                self._respawn_counters[attacker.id] = constants.RESPAWN_DELAY_TICKS
                self._drop_snake_pellets(attacker)
        collision.clamp_inside_world(self.snakes.values())
        self._maintain_pellet_population()

    def leaderboard(self) -> List[dict]:
        entries = sorted(
            [snake for snake in self.snakes.values()], key=lambda s: s.length, reverse=True
        )
        return [
            {"id": snake.id, "name": snake.name, "score": round(snake.length, 1)}
            for snake in entries[:10]
        ]

    def visible_pellets(self, snake: Snake) -> Iterable[Pellet]:
        head, _ = snake.head_circle()
        for pellet in self.pellets.values():
            if head.distance_to(pellet.position) <= constants.SNAPSHOT_RADIUS:
                yield pellet

    def visible_snakes(self, snake: Snake) -> Iterable[Snake]:
        head, _ = snake.head_circle()
        radius = constants.SNAPSHOT_RADIUS
        radius_sq = radius * radius
        for other in self.snakes.values():
            if not other.alive:
                continue
            other_head, _ = other.head_circle()
            if (other_head.x - head.x) ** 2 + (other_head.y - head.y) ** 2 <= radius_sq:
                yield other

    def snapshot_for(self, snake: Snake) -> str:
        return protocol.encode_snapshot(
            tick=self.tick,
            snakes=self.visible_snakes(snake),
            pellets=self.visible_pellets(snake),
            leaderboard=self.leaderboard(),
        )


async def run_world(world: World, tick_rate: int, broadcast_cb) -> None:
    """Run the simulation loop and invoke ``broadcast_cb`` every tick."""

    tick_interval = 1.0 / tick_rate
    while True:
        world.update()
        await broadcast_cb()
        await asyncio.sleep(tick_interval)
