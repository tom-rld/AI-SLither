"""Pygame based renderer for the game client."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import pygame

from .entities import PelletEntity
from .interpolation import InterpolatedSnake


class Renderer:
    """Responsible for all drawing tasks."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 18)
        self.leaderboard_font = pygame.font.SysFont("arial", 16)
        self.background_color = (20, 24, 28)
        self.world_color = (40, 46, 52)
        self.pellet_color = (255, 200, 90)

    def clear(self) -> None:
        self.screen.fill(self.background_color)

    def draw_world_bounds(self, center: Tuple[float, float], world_radius: float, camera: Tuple[float, float]) -> None:
        screen_center = (self.screen.get_width() / 2, self.screen.get_height() / 2)
        rel_x = center[0] - camera[0]
        rel_y = center[1] - camera[1]
        pygame.draw.circle(
            self.screen,
            self.world_color,
            (int(screen_center[0] + rel_x), int(screen_center[1] + rel_y)),
            int(world_radius),
            width=2,
        )

    def draw_pellets(self, pellets: Iterable[PelletEntity], camera: Tuple[float, float]) -> None:
        cx, cy = camera
        center = (self.screen.get_width() / 2, self.screen.get_height() / 2)
        for pellet in pellets:
            x = int(center[0] + (pellet.x - cx))
            y = int(center[1] + (pellet.y - cy))
            pygame.draw.circle(self.screen, self.pellet_color, (x, y), 3)

    def draw_snakes(self, snakes: Iterable[InterpolatedSnake], camera: Tuple[float, float]) -> None:
        cx, cy = camera
        center_x = self.screen.get_width() / 2
        center_y = self.screen.get_height() / 2
        for snake in snakes:
            for index, segment in enumerate(snake.segments):
                radius = 12 if index == 0 else 10
                x = int(center_x + (segment.x - cx))
                y = int(center_y + (segment.y - cy))
                pygame.draw.circle(self.screen, snake.color, (x, y), radius)
            if snake.segments:
                label = self.font.render(snake.name, True, (255, 255, 255))
                head = snake.segments[0]
                x = int(center_x + (head.x - cx))
                y = int(center_y + (head.y - cy) - 20)
                self.screen.blit(label, (x - label.get_width() / 2, y))

    def draw_leaderboard(self, entries: List[dict]) -> None:
        x = self.screen.get_width() - 200
        y = 20
        title = self.leaderboard_font.render("Leaderboard", True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        y += 24
        for index, entry in enumerate(entries):
            text = f"{index + 1}. {entry['name']} - {entry['score']}"
            surface = self.leaderboard_font.render(text, True, (220, 220, 220))
            self.screen.blit(surface, (x, y))
            y += 18

    def present(self) -> None:
        pygame.display.flip()
