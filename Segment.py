import pygame
from Object import Object

class Segment(Object):
    def __init__(self, x, y, w, h, filePath, speed):
        super().__init__(x, y, w, h, filePath)
        self.speed = speed

    def update(self, targetPos=None, head_speed=0):
        pass
