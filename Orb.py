import pygame
from Object import Object

SCORE_INCREMENT = 0.1

class Orb(Object):
    def __init__(self,x,y,r,filePath):
        super().__init__(x,y,r,r,filePath)

    def update(self, snakes):
        for snake in snakes:
            if self.rect.colliderect(snake.rect):
                snake.score += SCORE_INCREMENT
                return True
        return False
