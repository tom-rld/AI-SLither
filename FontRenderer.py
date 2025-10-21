import pygame

class FontRenderer:
    def __init__(self):
        self.color = (255,255,0)
        self.size = 80
        self.font = pygame.font.Font("fonts/Minecraft.ttf",self.size)


    def renderFont(self,window,score):
        text = self.font.render("Score: " + str(score),True,self.color)
        window.blit(text,(0,0))