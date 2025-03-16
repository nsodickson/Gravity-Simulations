import pygame
from pygame.locals import *
from pygame.math import Vector2
from utils import *
from typing import Self


class Body(pygame.sprite.Sprite):
    def __init__(self: Self, 
                 init_x: float, 
                 init_y: float, 
                 mass: float, 
                 radius: float, 
                 visual_radius: float=None, 
                 init_vel_x: float=0, 
                 init_vel_y: float=0, 
                 color: tuple[int, int, int]=RED) -> None:
        super(Body, self).__init__()
        
        self.pos = Vector2(init_x, init_y)
        self.vel = Vector2(init_vel_x, init_vel_y)
        self.prev_acc = Vector2(0, 0)
        self.acc = Vector2(0, 0)

        self.ke = 0
        self.pe = 0
        self.p = 0

        self.radius = radius
        if visual_radius is None:
            self.visual_radius = math.ceil(radius)
            self.visual_mode = "AUTOMATIC"  # Automatic mode changes the visual radius when the radius is changed
        else:
            self.visual_radius = visual_radius
            self.visual_mode = "MANUAL"  # Manual mode locks the visual radius in place, even if the radius is changed (not recommended)
        self.mass = mass

        self.color = color
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, self.color, (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(self.pos.x - self.visual_radius, self.pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setPos(self: Self, pos: Vector2) -> None:
        self.pos = pos
        self.rect.update(pos.x - self.visual_radius, pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setVel(self: Self, vel: Vector2) -> None:
        self.vel = vel
    
    def setPrevAcc(self: Self, acc: Vector2) -> None:
        self.prev_acc = acc
    
    def setAcc(self: Self, acc: Vector2) -> None:
        self.acc = acc
    
    def setRadius(self: Self, radius: float) -> None:
        self.radius = radius
        if self.visual_mode == "AUTOMATIC":
            self.visual_radius = math.ceil(radius)
        self.image = pygame.Surface((self.visual_radius * 2, self.visual_radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, self.color, (self.visual_radius, self.visual_radius), self.visual_radius)
        self.rect = pygame.Rect(self.pos.x - self.visual_radius, self.pos.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
    
    def setMass(self: Self, mass: float) -> None:
        self.mass = mass

    def getP(self: Self) -> Vector2:
        return self.mass * self.vel
    
    def getKE(self: Self) -> float:
        return 0.5 * self.mass * self.vel.magnitude() ** 2
    
    def setPE(self: Self, pe: float) -> None:
        self.pe = pe
    
    def getPE(self: Self) -> float:
        return self.pe
    
    def update(self: Self, dt: float, reset: bool=True) -> None:
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
            self.setAcc(Vector2(0, 0))
            self.setPE(0)

        self.rect.update(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def stop(self: Self) -> None:
        self.setVel(Vector2(0, 0))
        self.setPrevAcc(Vector2(0, 0))
        self.setAcc(Vector2(0, 0))
        self.setPE(0)
    
    def draw(self: Self, window: pygame.Surface) -> None:
        window.blit(self.image, (self.rect.x, self.rect.y))
    
    def draw_acc(self: Self, window: pygame.Surface, scale: float=10, max_length: float=500, color: tuple[int, int, int]=BLUE) -> None:
        visual_acc = self.acc * scale
        if visual_acc.magnitude() > max_length:
            visual_acc.scale_to_length(max_length)
        pygame.draw.line(window, color, self.pos, self.pos + visual_acc)

    def draw_vel(self: Self, window: pygame.Surface, scale: float=1, max_length: float=500, color: tuple[int, int, int]=GREEN) -> None:
        visual_vel = self.vel * scale
        if visual_vel.magnitude() > max_length:
            visual_vel.scale_to_length(max_length)
        pygame.draw.line(window, color, self.pos, self.pos + visual_vel)
