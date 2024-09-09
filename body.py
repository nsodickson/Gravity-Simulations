import pygame
from pygame.locals import *
from utils import *

class Body(pygame.sprite.Sprite):
    def __init__(self, x, y, mass, radius, visual_radius=None):
        super(Body, self).__init__()
        
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.acc = pygame.Vector2(0, 0)

        self.ke = 0
        self.pe = 0
        self.p = 0

        self.radius = radius
        if visual_radius is None:
            self.visual_radius = math.ceil(radius)
            self.visual_mode = "AUTOMATIC"
        else:
            self.visual_radius = visual_radius
            self.visual_mode = "MANUAL"
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
    
    def setAcc(self, acc):
        self.acc = acc
    
    def setRadius(self, radius):
        self.radius = radius
        if self.visual_mode == "AUTOMATIC":
            self.visual_radius = math.ceil(radius)
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, (255, 0, 0), (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(self.pos.x - self.visual_radius, self.pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setMass(self, mass):
        self.mass = mass
    
    def setPE(self, pe):
        self.pe = pe

    def getKE(self):
        return 0.5 * self.mass * self.vel.magnitude() ** 2
    
    def getPE(self):
        return self.pe
    
    def getP(self):
        return self.mass * self.vel.magnitude()
    
    def update(self, dt, reset=False):
        self.vel += self.acc * dt

        # Treat update as a progression of the simulation and automatically reset acceleration
        if reset:
            self.setAcc(pygame.Vector2(0, 0))
            self.setPE(0)

        if self.vel.magnitude() > C:
            self.vel.scale_to_length(C)

        self.pos += self.vel * dt
        self.rect.update(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def stop(self):
        self.vel = pygame.Vector2(0, 0)
    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))
    
    def draw_acc(self, window, scale=10, max_length=500):
        visual_acc = self.acc * scale
        if visual_acc.magnitude() > max_length:
            visual_acc.scale_to_length(max_length)
        pygame.draw.line(window, BLUE, self.pos, self.pos + visual_acc)
    
    def draw_vel(self, window, scale=1, max_length=500):
        visual_vel = self.vel * scale
        if visual_vel.magnitude() > max_length:
            visual_vel.scale_to_length(max_length)
        pygame.draw.line(window, GREEN, self.pos, self.pos + visual_vel)
