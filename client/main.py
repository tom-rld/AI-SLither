"""Entry point for the pygame based client."""

from __future__ import annotations

import argparse
import asyncio
import pygame

from .entities import EntityStore
from .input import InputManager
from .interpolation import Interpolator
from .network import NetworkClient
from .render import Renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI-Slither client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--name", default="Player", help="Player nickname")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    return parser.parse_args()


async def run_client(args: argparse.Namespace) -> None:
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("AI-Slither")
    renderer = Renderer(screen)
    clock = pygame.time.Clock()

    uri = f"ws://{args.host}:{args.port}"
    network = NetworkClient(uri, args.name)
    welcome = await network.connect()

    store = EntityStore()
    interpolator = Interpolator()
    input_manager = InputManager()

    player_id = welcome.get("id")
    world_radius = welcome.get("worldRadius", 60_000)
    leaderboard: list[dict] = []
    camera = (0.0, 0.0)
    snapshot_task = asyncio.create_task(network.next_snapshot())
    running = True

    while running:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        mouse_pos = pygame.mouse.get_pos()
        boost = pygame.mouse.get_pressed(num_buttons=3)[0]
        input_state = input_manager.update(mouse_pos, screen.get_size(), boost)
        await network.send_input(input_state)

        if snapshot_task.done():
            snapshot = snapshot_task.result()
            if snapshot.get("type") == "snapshot":
                store.update_from_snapshot(snapshot)
                leaderboard = snapshot.get("leaderboard", [])
                player = store.snakes.get(player_id)
                if player and player.segments:
                    head = player.segments[0]
                    camera = (head.x, head.y)
            elif snapshot.get("type") == "disconnect":
                running = False
            snapshot_task = asyncio.create_task(network.next_snapshot())

        renderer.clear()
        renderer.draw_world_bounds((0, 0), world_radius, camera)
        renderer.draw_pellets(store.pellets.values(), camera)
        snakes_to_draw = interpolator.build(store.snakes)
        renderer.draw_snakes(snakes_to_draw, camera)
        renderer.draw_leaderboard(leaderboard)
        renderer.present()
        await asyncio.sleep(0)

    await network.close()
    pygame.quit()


def main() -> None:
    args = parse_args()
    asyncio.run(run_client(args))


if __name__ == "__main__":
    main()
