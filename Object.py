import pygame

class Object:
    def __init__(self, x,y,w,h,filePath):
        self.rect = pygame.Rect(x, y, w, h)
        image = pygame.image.load(filePath)
        self.texturePath = filePath
        self.texture = pygame.transform.scale(image, (w, h))

    def draw(self, window, camera):
        window.blit(self.texture, camera.translate(self.rect.x, self.rect.y))

    def update(self):
        pass