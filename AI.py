from Snake import Snake
import math
import random
import pygame

SAFE_DISTANCE = 150
MAP_MARGIN = 120
STUCK_TIME = 1500
PANIC_DURATION = 700

class AI(Snake):
    def __init__(self, x, y, w, h, filePath):
        super().__init__(x, y, w, h, filePath)
        self.target_orb = None
        self.randomness_timer = 0
        self.random_offset = [0, 0]
        self.last_pos = (x, y)
        self.stuck_start = None
        self.in_panic = False
        self.panic_timer = 0
        self.time_near_border = 0
        self.last_stable_dir = [1, 0]  # mémoire directionnelle
        self.border_side = None        # "CW" ou "CCW" (rotation sens horaire ou anti)

    def update(self, orbs, snakes, map_center=(0, 0), map_radius=2000):
        self.think(orbs, snakes, map_center, map_radius)
        return super().update(snakes)

    def think(self, orbs, snakes, map_center, map_radius):
        now = pygame.time.get_ticks()

        # Panic move actif ?
        if self.in_panic:
            if now - self.panic_timer < PANIC_DURATION:
                return
            self.in_panic = False

        # Choisir cible
        if not self.target_orb or self.target_orb not in orbs:
            self.target_orb = self.choose_best_orb(orbs)

        move_dir = [0, 0]
        if self.target_orb:
            move_dir = [
                self.target_orb.rect.centerx - self.rect.centerx,
                self.target_orb.rect.centery - self.rect.centery
            ]

        # --- éviter serpents ---
        avoid_x, avoid_y = 0, 0
        for snake in snakes:
            if snake == self:
                continue
            dx = snake.rect.centerx - self.rect.centerx
            dy = snake.rect.centery - self.rect.centery
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < SAFE_DISTANCE and dist > 0:
                avoid_x -= dx / dist
                avoid_y -= dy / dist
        move_dir[0] += avoid_x * 3
        move_dir[1] += avoid_y * 3

        # --- random offset ---
        if now - self.randomness_timer > 1800:
            self.random_offset = [random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)]
            self.randomness_timer = now
        move_dir[0] += self.random_offset[0]
        move_dir[1] += self.random_offset[1]

        # --- gestion de la limite circulaire ---
        dx = self.rect.centerx - map_center[0]
        dy = self.rect.centery - map_center[1]
        dist_from_center = math.sqrt(dx * dx + dy * dy)
        dist_from_edge = map_radius - dist_from_center

        if dist_from_edge < MAP_MARGIN:
            self.time_near_border += 16

            # vecteur radial
            nx, ny = -dx / (dist_from_center + 1e-6), -dy / (dist_from_center + 1e-6)
            # tangente stable
            if self.border_side is None:
                self.border_side = "CW" if random.random() < 0.5 else "CCW"
            tangent = [ny, -nx] if self.border_side == "CW" else [-ny, nx]

            # interpolation douce (évite le switch brutal)
            factor = (MAP_MARGIN - dist_from_edge) / MAP_MARGIN
            move_dir[0] = self.lerp(move_dir[0], tangent[0] + nx * factor, 0.3)
            move_dir[1] = self.lerp(move_dir[1], tangent[1] + ny * factor, 0.3)

            # si reste trop longtemps sur le bord → dash vers le centre
            if self.time_near_border > 2500:
                self.direction = [nx, ny]
                self.boosting = True
                self.in_panic = True
                self.panic_timer = now
                self.time_near_border = 0
                self.border_side = None
                return
        else:
            self.time_near_border = 0
            self.border_side = None

        # --- normalisation et inertie ---
        length = math.sqrt(move_dir[0] ** 2 + move_dir[1] ** 2)
        if length != 0:
            move_dir[0] /= length
            move_dir[1] /= length

            # Inertie : mélange avec ancienne direction
            move_dir[0] = self.lerp(self.last_stable_dir[0], move_dir[0], 0.2)
            move_dir[1] = self.lerp(self.last_stable_dir[1], move_dir[1], 0.2)

            # Renormaliser après inertie
            n = math.sqrt(move_dir[0] ** 2 + move_dir[1] ** 2)
            move_dir[0] /= n
            move_dir[1] /= n
            self.last_stable_dir = move_dir.copy()
        else:
            move_dir = self.last_stable_dir.copy()

        self.direction = move_dir

        # --- boost logique ---
        close_enemy = any(
            math.hypot(s.rect.centerx - self.rect.centerx, s.rect.centery - self.rect.centery) < 200
            for s in snakes if s != self
        )

        if self.target_orb and not close_enemy and self.can_boost:
            dist_to_orb = math.hypot(
                self.target_orb.rect.centerx - self.rect.centerx,
                self.target_orb.rect.centery - self.rect.centery
            )
            self.boosting = dist_to_orb < 250
        elif close_enemy:
            self.boosting = True
        else:
            self.boosting = False

        # --- anti-stuck ---
        dx = self.rect.x - self.last_pos[0]
        dy = self.rect.y - self.last_pos[1]
        distance_moved = math.sqrt(dx * dx + dy * dy)
        if distance_moved < 3:
            if self.stuck_start is None:
                self.stuck_start = now
            elif now - self.stuck_start > STUCK_TIME:
                self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
                n = math.sqrt(self.direction[0] ** 2 + self.direction[1] ** 2)
                self.direction[0] /= n
                self.direction[1] /= n
                self.boosting = True
                self.in_panic = True
                self.panic_timer = now
                self.stuck_start = None
        else:
            self.stuck_start = None

        self.last_pos = (self.rect.x, self.rect.y)

    def choose_best_orb(self, orbs):
        if not orbs:
            return None
        best_orb, best_value = None, float('inf')
        for orb in orbs:
            # ⚠️ Ignore les orbes trop proches du bord (hors map)
            ox, oy = orb.rect.centerx, orb.rect.centery
            dist = math.sqrt(ox ** 2 + oy ** 2)
            if dist > 1900:  # proche du bord du cercle (map_radius=2000)
                continue

            dx = ox - self.rect.centerx
            dy = oy - self.rect.centery
            distance = math.sqrt(dx ** 2 + dy ** 2)
            value = distance / (orb.rect.width + 1)

            if value < best_value:
                best_value = value
                best_orb = orb
        return best_orb

    def lerp(self, a, b, t):
        return a + (b - a) * t
