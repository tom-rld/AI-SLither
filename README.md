# AI-Slither

AI-Slither is a light-weight reimplementation of the classic Slither.io gameplay.
The project is split into an asyncio based authoritative server and a pygame
client. Both components communicate through a JSON protocol transported over
WebSockets.

## Features

- Authoritative physics simulation with snake growth, boosting and collisions
- World populated with pellets that can be collected or dropped while boosting
- Automatic respawn handling and a live leaderboard of the top snakes
- Pygame renderer with mouse based input and on-screen leaderboard

## Project layout

```
server/
    main.py          # Starts the asyncio websocket server
    world.py         # Holds the authoritative simulation state
    snake.py         # Snake entity logic including boosting & growth
    pellet.py        # Pellet entity definition
    collision.py     # Collision detection helpers
    protocol.py      # JSON serialisation helpers
    utils.py         # Math primitives such as Vec2
    constants.py     # Gameplay tuning constants
client/
    main.py          # Pygame bootstrap and main loop
    network.py       # Async websocket client
    render.py        # Drawing routines
    entities.py      # Local representations of remote entities
    interpolation.py # Frame interpolation stub
    input.py         # Mouse input mapping
```

## Getting started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The project depends on `pygame` and `websockets` in addition to the Python
   standard library.

2. Start the server:

   ```bash
   python -m server.main --host 0.0.0.0 --port 8765
   ```

3. Launch the client (in a separate terminal):

   ```bash
   python -m client.main --host 127.0.0.1 --port 8765 --name Alice
   ```

## Controls

- Move the mouse to steer your snake.
- Hold the left mouse button to boost at the cost of mass.
- Close the window to exit the client.

## Development notes

The project is intentionally structured to keep server and client logic
separated. The server is authoritative and applies all physics, collision and
leaderboard rules. Clients simply render the snapshots that are streamed by the
server. This allows the logic to be unit tested in isolation from the renderer
and enables future extensions such as bots or matchmaking services.
