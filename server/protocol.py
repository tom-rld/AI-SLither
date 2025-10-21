"""JSON protocol helpers for the websocket transport."""

from __future__ import annotations

import json
from typing import Iterable, List

from .snake import Snake
from .pellet import Pellet


def parse_client_message(message: str) -> dict:
    """Parse a raw client ``message`` into a Python dictionary."""

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
        raise ValueError("Invalid client message") from exc
    if not isinstance(payload, dict):
        raise ValueError("Client message must be a JSON object")
    return payload


def encode_snapshot(tick: int, snakes: Iterable[Snake], pellets: Iterable[Pellet], leaderboard: List[dict]) -> str:
    """Encode a world snapshot for broadcasting to clients."""

    snapshot = {
        "type": "snapshot",
        "tick": tick,
        "snakes": [snake.to_snapshot() for snake in snakes],
        "pellets": [pellet.to_dict() for pellet in pellets],
        "leaderboard": leaderboard,
    }
    return json.dumps(snapshot)


def encode_welcome(snake: Snake, world_radius: float) -> str:
    """Encode the welcome payload sent upon connection."""

    return json.dumps(
        {
            "type": "welcome",
            "id": snake.id,
            "name": snake.name,
            "color": snake.color,
            "worldRadius": world_radius,
        }
    )
