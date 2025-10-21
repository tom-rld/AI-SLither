"""Websocket networking client."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from .input import InputState


class NetworkClient:
    """Asynchronous websocket client that exchanges snapshots with the server."""

    def __init__(self, uri: str, name: str) -> None:
        self.uri = uri
        self.name = name
        self.websocket: Optional[WebSocketClientProtocol] = None
        self._incoming: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._receiver_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> Dict[str, Any]:
        self.websocket = await websockets.connect(self.uri)
        await self._send_json({"type": "join", "name": self.name})
        welcome = await self._recv_json()
        self._receiver_task = asyncio.create_task(self._receiver_loop())
        return welcome

    async def _receiver_loop(self) -> None:
        assert self.websocket is not None
        try:
            async for message in self.websocket:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    continue
                await self._incoming.put(payload)
        finally:
            await self._incoming.put({"type": "disconnect"})

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        if self.websocket is None:
            raise RuntimeError("Client is not connected")
        await self.websocket.send(json.dumps(payload))

    async def _recv_json(self) -> Dict[str, Any]:
        if self.websocket is None:
            raise RuntimeError("Client is not connected")
        message = await self.websocket.recv()
        return json.loads(message)

    async def send_input(self, state: InputState) -> None:
        await self._send_json({"type": "input", "angle": state.angle, "boost": state.boost})

    async def next_snapshot(self) -> Dict[str, Any]:
        payload = await self._incoming.get()
        return payload

    async def close(self) -> None:
        if self.websocket is not None:
            await self.websocket.close()
        if self._receiver_task is not None:
            await self._receiver_task
