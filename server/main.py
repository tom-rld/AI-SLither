"""Entry point for the asyncio based game server."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Dict

import websockets
from websockets.server import WebSocketServerProtocol

from . import constants, protocol
from .world import World


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class GameServer:
    """High level orchestration of the world simulation and websocket IO."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.world = World()
        self.clients: Dict[int, WebSocketServerProtocol] = {}
        self._broadcast_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the websocket server and the world update loop."""

        async with websockets.serve(self._handle_client, self.host, self.port):
            logging.info("Server listening on %s:%s", self.host, self.port)
            await self._run_game_loop()

    async def _run_game_loop(self) -> None:
        tick_interval = 1.0 / constants.TICK_RATE
        while True:
            self.world.update()
            await self._broadcast_snapshots()
            await asyncio.sleep(tick_interval)

    async def _broadcast_snapshots(self) -> None:
        if not self.clients:
            return
        async with self._broadcast_lock:
            disconnected = []
            for snake_id, ws in self.clients.items():
                snake = self.world.snakes.get(snake_id)
                if not snake:
                    disconnected.append(snake_id)
                    continue
                try:
                    payload = self.world.snapshot_for(snake)
                    await ws.send(payload)
                except Exception:  # pragma: no cover - we simply drop failed clients
                    logging.exception("Failed to send snapshot to client %s", snake_id)
                    disconnected.append(snake_id)
            for snake_id in disconnected:
                ws = self.clients.pop(snake_id, None)
                if ws:
                    await ws.close()
                self.world.remove_snake(snake_id)

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        name = await self._receive_join_name(websocket)
        snake = self.world.add_snake(name)
        self.clients[snake.id] = websocket
        await websocket.send(protocol.encode_welcome(snake, constants.WORLD_RADIUS))
        logging.info("Player %s connected as snake %s", name, snake.id)
        try:
            async for message in websocket:
                try:
                    payload = protocol.parse_client_message(message)
                except ValueError:
                    continue
                if payload.get("type") == "input":
                    angle = float(payload.get("angle", 0.0))
                    boost = bool(payload.get("boost", False))
                    self.world.set_player_input(snake.id, angle, boost)
        except websockets.ConnectionClosed:
            logging.info("Client %s disconnected", snake.id)
        finally:
            self.clients.pop(snake.id, None)
            self.world.remove_snake(snake.id)

    async def _receive_join_name(self, websocket: WebSocketServerProtocol) -> str:
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except asyncio.TimeoutError:
            return f"Player{len(self.clients) + 1}"
        try:
            payload = protocol.parse_client_message(message)
        except ValueError:
            return f"Player{len(self.clients) + 1}"
        if payload.get("type") == "join" and "name" in payload:
            name = str(payload["name"]).strip()
            if name:
                return name[:16]
        return f"Player{len(self.clients) + 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI-Slither server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = GameServer(args.host, args.port)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
