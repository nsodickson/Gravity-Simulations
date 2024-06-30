import pygame
from pygame.locals import *
from utils import *

class Body(pygame.sprite.Sprite):
    def __init__(self, x, y, mass, radius):
        super(Body, self).__init__()
        
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)

        self.radius = radius
        self.visual_radius = math.ceil(radius)
        self.mass = mass

        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, (255, 0, 0), (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(x - self.visual_radius, y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
        
        self.clicked = False
    
    def setPos(self, pos):
        self.pos = pos
        self.rect.update(pos.x - self.visual_radius, pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setVel(self, vel):
        self.vel = vel
    
    def setRadius(self, radius):
        self.radius = radius
        self.visual_radius = math.ceil(radius)
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, (255, 0, 0), (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(self.pos.x - self.visual_radius, self.pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setMass(self, mass):
        self.mass = mass
    
    def update(self, dt):
        self.pos += self.vel * dt
        self.rect.update(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def stop(self):
        self.vel = pygame.Vector2(0, 0)
    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))