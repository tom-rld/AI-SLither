import pygame
from Player import Player
from Orb import Orb
from Camera import Camera
from AI import AI
from FontRenderer import FontRenderer
import random
import math


START_W = 50
START_H = 50

PLAYER_START_X = 0
PLAYER_START_Y = 0
PLAYER_TEXTURE = "textures/body_red.png"

MAX_ORB_SIZE = 40

FPS = 60
NUM_ORBS = 200
NUM_AI = 3


class MainGame:
    def __init__(self):
        pygame.init()

        self.winDims = (1000, 700)
        self.window = pygame.display.set_mode(self.winDims)
        self.winColor = (75, 75, 75)
        self.quit = False
        self.clock = pygame.time.Clock()

        self.camera = Camera(PLAYER_START_X, PLAYER_START_Y, (START_W, START_H), self.winDims)

        self.textures = [
            "textures/blue_orb.png",
            "textures/green_orb.png",
            "textures/purple_orb.png",
            "textures/red_orb.png",
            "textures/yellow_orb.png",
            "textures/orange_orb.png"
        ]

        self.player = Player(PLAYER_START_X, PLAYER_START_Y, START_W, START_H, random.choice(self.textures), self.winDims)
        self.orbs = []
        self.snakes = []
        self.fontRenderer = FontRenderer()

        self.game_over = False
        self.font = pygame.font.SysFont("arial", 48, bold=True)

    def init(self):
        # --- configuration de la MAP LIMITÉE ---
        self.map_radius = 2000  # rayon de la map (ajuste selon ton goût)
        self.map_center = (0, 0)  # centre de la map (origine)
        print(f"🌍 Map limitée initialisée (rayon = {self.map_radius})")

        # --- génération des orbes (dans le cercle rouge) ---
        for i in range(NUM_ORBS):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.95)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            randR = random.randint(10, MAX_ORB_SIZE)
            randTexture = random.choice(self.textures)
            newOrb = Orb(randX, randY, randR, randTexture)
            self.orbs.append(newOrb)

        # --- ajout du joueur ---
        self.snakes.append(self.player)

        # --- génération des IA (dans la zone autorisée) ---
        for i in range(NUM_AI):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.9)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            randTexture = random.choice(self.textures)
            newAI = AI(randX, randY, START_W, START_H, randTexture)
            self.snakes.append(newAI)

        # --- lancement du jeu ---
        self.play()

    def play(self):
        while not self.quit:
            self.update()
            self.render()

    def update(self):
        """Met à jour tout le jeu (player, IA, collisions, orbes, limites de map, etc.)."""
        self.clock.tick(FPS)

        # --- Gestion des événements ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True

        keys = pygame.key.get_pressed()
        if self.game_over:
            if keys[pygame.K_r]:
                self.reset_game()
            return

        # --- Mise à jour des orbes ---
        for orb in self.orbs[:]:
            if orb.update(self.snakes):
                self.orbs.remove(orb)

        # --- Respawn d'orbes manquantes ---
        TARGET_ORB_COUNT = 250  # densité cible à maintenir
        self.last_orb_respawn = getattr(self, "last_orb_respawn", 0)
        if pygame.time.get_ticks() - self.last_orb_respawn > 1000:  # toutes les 1 seconde
            if len(self.orbs) < TARGET_ORB_COUNT:
                for _ in range(min(5, TARGET_ORB_COUNT - len(self.orbs))):  # max 5 par tick
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(0, self.map_radius * 0.95)
                    randX = int(self.map_center[0] + radius * math.cos(angle))
                    randY = int(self.map_center[1] + radius * math.sin(angle))
                    randR = random.randint(10, MAX_ORB_SIZE)
                    randTexture = random.choice(self.textures)
                    self.orbs.append(Orb(randX, randY, randR, randTexture))
                self.last_orb_respawn = pygame.time.get_ticks()

        # --- Mise à jour des serpents ---
        player_died = False
        dead_ai = []

        for snake in self.snakes[:]:
            died = snake.update(self.orbs, self.snakes)

            # --- Vérifie la limite de la MAP ---
            dx = snake.rect.centerx - self.map_center[0]
            dy = snake.rect.centery - self.map_center[1]
            dist_from_center = math.sqrt(dx**2 + dy**2)
            out_of_bounds = dist_from_center > self.map_radius

            if died or out_of_bounds:
                # Crée des orbes à l’endroit de la mort
                startX, startY = snake.rect.x, snake.rect.y
                size = START_W
                randTexture = random.choice(self.textures)
                self.orbs.append(Orb(startX, startY, size, randTexture))

                for segment in snake.segments:
                    sx, sy = segment.rect.x, segment.rect.y
                    randTexture = random.choice(self.textures)
                    self.orbs.append(Orb(sx, sy, size, randTexture))

                snake.segments.clear()
                self.snakes.remove(snake)

                if snake is self.player:
                    player_died = True
                    break
                else:
                    dead_ai.append(snake)

        if player_died:
            self.game_over = True
            return

        # --- Respawn des IA mortes à l'intérieur du cercle rouge ---
        for _ in dead_ai:
            randTexture = random.choice(self.textures)
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.9)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            new_ai = AI(randX, randY, START_W, START_H, randTexture)
            self.snakes.append(new_ai)

        # --- Maintenir un maximum de 5 IA ---
        ai_list = [s for s in self.snakes if isinstance(s, AI)]
        while len(ai_list) < 5:
            randTexture = random.choice(self.textures)
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.9)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            new_ai = AI(randX, randY, START_W, START_H, randTexture)
            self.snakes.append(new_ai)
            ai_list.append(new_ai)

        if len(ai_list) > 5:
            for extra_ai in ai_list[5:]:
                self.snakes.remove(extra_ai)

        # --- Mise à jour de la caméra ---
        self.camera.update(
            self.player.rect.centerx,
            self.player.rect.centery,
            map_center=self.map_center,
            map_radius=self.map_radius
        )

    def reset_game(self):
        """Réinitialise complètement une partie et recentre la caméra."""
        self.orbs.clear()
        self.snakes.clear()

        # --- Recrée le joueur ---
        self.player = Player(0, 0, START_W, START_H, random.choice(self.textures), self.winDims)
        self.snakes.append(self.player)

        # --- Recrée les IA ---
        for _ in range(NUM_AI):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.9)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            randTexture = random.choice(self.textures)
            self.snakes.append(AI(randX, randY, START_W, START_H, randTexture))

        # --- Recrée les orbes dans la map circulaire ---
        for i in range(NUM_ORBS):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, self.map_radius * 0.95)
            randX = int(self.map_center[0] + radius * math.cos(angle))
            randY = int(self.map_center[1] + radius * math.sin(angle))
            randR = random.randint(10, MAX_ORB_SIZE)
            randTexture = random.choice(self.textures)
            self.orbs.append(Orb(randX, randY, randR, randTexture))

        self.game_over = False

        # --- Recentrer la caméra ---
        self.camera.update(
            self.player.rect.centerx,
            self.player.rect.centery,
            map_center=self.map_center,
            map_radius=self.map_radius
        )

    def render(self):
        """Affiche tout : fond, bord de map, orbes, serpents, overlay Game Over."""
        self.window.fill((20, 20, 20))

        # --- Bordure rouge de la map ---
        map_screen_center = self.camera.translate(self.map_center[0], self.map_center[1])
        pygame.draw.circle(self.window, (255, 0, 0), map_screen_center, self.map_radius, 4)

        # --- Orbes ---
        for orb in self.orbs:
            orb.draw(self.window, self.camera)

        # --- Serpents ---
        for snake in self.snakes:
            snake.draw(self.window, self.camera)

        # --- Overlay Game Over ---
        if self.game_over:
            overlay = pygame.Surface(self.winDims, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.window.blit(overlay, (0, 0))
            text = self.font.render("GAME OVER - Appuie sur R pour rejouer", True, (255, 80, 80))
            rect = text.get_rect(center=(self.winDims[0] // 2, self.winDims[1] // 2))
            self.window.blit(text, rect)

        pygame.display.update()
