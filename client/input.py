"""Translate local input into commands for the server."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class InputState:
    """Represents the control state to send to the server."""

    angle: float
    boost: bool


class InputManager:
    """Calculate the desired movement vector from the mouse position."""

    def __init__(self) -> None:
        self._last_state = InputState(0.0, False)

    def update(self, mouse_pos: Tuple[int, int], viewport_size: Tuple[int, int], boost_button: bool) -> InputState:
        center_x = viewport_size[0] / 2
        center_y = viewport_size[1] / 2
        dx = mouse_pos[0] - center_x
        dy = mouse_pos[1] - center_y
        angle = math.atan2(dy, dx)
        self._last_state = InputState(angle, boost_button)
        return self._last_state
