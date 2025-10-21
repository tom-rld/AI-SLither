import pygame
from Snake import Snake
from Segment import Segment

class Player(Snake):
    def __init__(self, x, y, w, h, filePath, winDims):
        super().__init__(x, y, w, h, filePath)
        self.winDims = winDims

    def update(self, orbs, snakes):
        self.calculateDirection()

        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0] and self.can_boost:
            self.boosting = True
        else:
            self.boosting = False

        return super().update(snakes)

    def calculateDirection(self):
        mousePos = pygame.mouse.get_pos()
        worldPos = (mousePos[0] - self.winDims[0] / 2 + self.rect.x,
                    mousePos[1] - self.winDims[1] / 2 + self.rect.y)

        self.direction = [worldPos[0] - self.rect.x, worldPos[1] - self.rect.y]
        length = (self.direction[0] ** 2 + self.direction[1] ** 2) ** 0.5
        if length != 0:
            self.direction[0] /= length
            self.direction[1] /= length
