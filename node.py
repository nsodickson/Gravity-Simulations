import pygame
from pygame.locals import *
from utils import *

class Node:
    def __init__(self, x, y, s, depth=100):
        self.bodies = []
        self.children = []
        self.s = s
        self.half_s = round(s / 2)
        self.rect = pygame.Rect(x, y, s, s)

        self.mass = 0
        self.weighted_pos = pygame.Vector2(0, 0)
        self.center = pygame.Vector2(x + self.half_s, y + self.half_s)

        self.internal = False
        self.depth = depth
    
    def addBody(self, body):
        self.mass += body.mass
        self.weighted_pos += body.pos * body.mass
        self.center = self.weighted_pos / self.mass

        if self.depth > 0:
            if self.internal:  # The node is an internal node
                idx = self.selectChild(body)
                self.children[idx].addBody(body)
            elif len(self.bodies) == 1:  # The node becomes registered as an internal node
                self.internal = True
                self.children = [Node(self.rect.x, self.rect.y, self.half_s, self.depth - 1),
                                 Node(self.rect.x + self.half_s, self.rect.y, self.half_s, self.depth - 1),
                                 Node(self.rect.x, self.rect.y + self.half_s, self.half_s, self.depth - 1),
                                 Node(self.rect.x + self.half_s, self.rect.y + self.half_s, self.half_s, self.depth - 1)]
                self.bodies.append(body)
                for body in self.bodies:
                    idx = self.selectChild(body)
                    self.children[idx].addBody(body)
                self.bodies.clear()
            else:  # The node is an external node
                self.bodies.append(body)
        else:  # The node is an external node
            self.bodies.append(body)

    def selectChild(self, body):
        if body.pos.x >= self.rect.x + self.half_s:
            if body.pos.y >= self.rect.y + self.half_s:
                return 3
            else:
                return 1
        else:
            if body.pos.y >= self.rect.y + self.half_s:
                return 2
            else:
                return 0
    
    def draw(self, window):
        pygame.draw.rect(window, (0, 0, 255), self.rect, 1)
        for child in self.children:
            child.draw(window)
    
    def drawGravity(self, window, pos, theta):
        if not self.internal:
            pygame.draw.rect(window, (0, 0, 255), self.rect, 1)
        elif self.s / dist(self.center, pos) > theta:
            for child in self.children:
                child.drawGravity(window, pos, theta)
        else:
            pygame.draw.rect(window, (0, 255, 0), self.rect, 1)
    
    def drawMerge(self, window, body):
        if body.rect.colliderect(self.rect):
            if not self.internal:
                pygame.draw.rect(window, (0, 0, 255), self.rect, 1)
            else:
                for child in self.children:
                    child.drawMerge(window, body)
        else:
            pygame.draw.rect(window, (0, 255, 0), self.rect, 1)
