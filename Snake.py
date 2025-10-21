import pygame
from Object import Object
from Segment import Segment
from collections import deque

MAX_CHECK_DISTANCE = 500

class Snake(Object):
    def __init__(self, x, y, w, h, filePath):
        super().__init__(x, y, w, h, filePath)

        # --- paramètres principaux ---
        self.prevScore = 0
        self.score = 100
        self.direction = [0, 0]
        self.segments = []

        # --- paramètres du boost ---
        self.boosting = False
        self.base_speed = 7
        self.boost_speed = 14

        # --- durée et recharge du boost ---
        self.boost_duration = 1500   # 1.5s
        self.boost_cooldown = 3000   # 3s
        self.boost_time_left = self.boost_duration
        self.last_boost_time = 0
        self.can_boost = True

        # --- historique de positions de la tête (trajectoire) ---
        self.trail = deque(maxlen=5000)
        self.spacing = int(w * 0.4)  # espacement entre segments (ajustable)

        # --- taille de base ---
        self.segment_size = w
        self.segment_growth = 20  # combien de points de score = un segment

    def update(self, snakes):
        current_time = pygame.time.get_ticks()

        # --- Gestion du boost (durée + recharge) ---
        if self.boosting and self.can_boost:
            self.boost_time_left -= 16
            if self.boost_time_left <= 0:
                self.boost_time_left = 0
                self.boosting = False
                self.can_boost = False
                self.last_boost_time = current_time
        else:
            if not self.can_boost and (current_time - self.last_boost_time >= self.boost_cooldown):
                self.can_boost = True
                self.boost_time_left = self.boost_duration

        # --- Déplacement ---
        current_speed = self.boost_speed if self.boosting and self.can_boost else self.base_speed
        if len(self.direction) == 2:
            self.rect.x += self.direction[0] * current_speed
            self.rect.y += self.direction[1] * current_speed

        # --- Enregistre la trace de la tête ---
        self.trail.appendleft((self.rect.x, self.rect.y))

        # --- Met à jour la taille du corps ---
        self.updateLength()

        # --- Fait suivre les segments le long de la trace ---
        self.updateSegments()

        # --- Vérifie les collisions (pas d’auto-collision) ---
        return self.checkCollision(snakes)

    def updateLength(self):
        """Ajoute des segments selon le score"""
        target_length = max(0, int(self.score / self.segment_growth))
        while len(self.segments) < target_length:
            # on place le segment à la fin de la trace, s’il y a assez de points
            index = (len(self.segments) + 1) * self.spacing
            if index < len(self.trail):
                pos = self.trail[index]
            else:
                pos = self.trail[-1] if self.trail else (self.rect.x, self.rect.y)
            newSegment = Segment(pos[0], pos[1], self.segment_size, self.segment_size, self.texturePath, self.base_speed)
            self.segments.append(newSegment)

    def updateSegments(self):
        """Chaque segment suit une distance fixe derrière la tête, même en boost"""
        if len(self.trail) < 2:
            return

        # calcul des distances cumulées sur la trail
        distances = [0]
        for i in range(1, len(self.trail)):
            x1, y1 = self.trail[i - 1]
            x2, y2 = self.trail[i]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            distances.append(distances[-1] + dist)

        total_length = distances[-1]

        for i, segment in enumerate(self.segments):
            target_dist = (i + 1) * self.spacing

            if target_dist > total_length:
                # si le serpent est plus court que le trail, place à la fin
                segment.rect.x, segment.rect.y = self.trail[-1]
                continue

            # cherche deux points du trail entre lesquels se trouve cette distance
            for j in range(1, len(distances)):
                if distances[j] >= target_dist:
                    ratio = (target_dist - distances[j - 1]) / (distances[j] - distances[j - 1] + 1e-6)
                    x1, y1 = self.trail[j - 1]
                    x2, y2 = self.trail[j]
                    segment.rect.x = x1 + (x2 - x1) * ratio
                    segment.rect.y = y1 + (y2 - y1) * ratio
                    break

    def checkCollision(self, snakes):
        """Collision uniquement avec les autres serpents"""
        for snake in snakes:
            if snake == self:
                continue

            dx = snake.rect.x - self.rect.x
            dy = snake.rect.y - self.rect.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > MAX_CHECK_DISTANCE:
                continue

            for segment in snake.segments:
                if self.rect.colliderect(segment.rect):
                    return True
        return False

    def draw(self, window, camera):
        # tête
        window.blit(self.texture, camera.translate(self.rect.x, self.rect.y))
        # segments
        for segment in self.segments:
            window.blit(segment.texture, camera.translate(segment.rect.x, segment.rect.y))
