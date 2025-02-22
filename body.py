import pygame
from pygame.locals import *
from utils import *

class Body(pygame.sprite.Sprite):
    def __init__(self, x, y, mass, radius, visual_radius=None, init_vel_x=0, init_vel_y=0, color=RED):
        super(Body, self).__init__()
        
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(init_vel_x, init_vel_y)
        self.prev_acc = pygame.Vector2(0, 0)
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

        self.color = color
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, self.color, (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(x - self.visual_radius, y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setPos(self, pos):
        self.pos = pos
        self.rect.update(pos.x - self.visual_radius, pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setVel(self, vel):
        self.vel = vel
    
    def setPrevAcc(self, acc):
        self.prev_acc = acc
    
    def setAcc(self, acc):
        self.acc = acc
    
    def setRadius(self, radius):
        self.radius = radius
        if self.visual_mode == "AUTOMATIC":
            self.visual_radius = math.ceil(radius)
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, self.color, (self.visual_radius, self.visual_radius), self.visual_radius)
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
        return self.mass * self.vel
    
    def update(self, dt, reset=True):
        # Euler Integration (Not Good)
        """
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        """
        
        # Half Step Velocity Verlet Integration (I Think???)
        self.vel += self.prev_acc * dt / 2
        self.pos += self.vel * dt + 0.5 * self.prev_acc * dt ** 2
        self.vel += self.acc * dt / 2
        self.setPrevAcc(self.acc)

        # Treat update as a progression of the simulation and automatically reset acceleration
        if reset:
            self.setAcc(pygame.Vector2(0, 0))
            self.setPE(0)

        self.rect.update(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def stop(self):
        self.setVel(pygame.Vector2(0, 0))
        self.setPrevAcc(pygame.Vector2(0, 0))
        self.setAcc(pygame.Vector2(0, 0))
        self.setPE(0)
    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))
    
    def draw_acc(self, window, scale=10, max_length=500, color=BLUE):
        visual_acc = self.acc * scale
        if visual_acc.magnitude() > max_length:
            visual_acc.scale_to_length(max_length)
        pygame.draw.line(window, color, self.pos, self.pos + visual_acc)
    
    def draw_vel(self, window, scale=1, max_length=500, color=GREEN):
        visual_vel = self.vel * scale
        if visual_vel.magnitude() > max_length:
            visual_vel.scale_to_length(max_length)
        pygame.draw.line(window, color, self.pos, self.pos + visual_vel)
