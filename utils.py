import pygame
from pygame.locals import *
import math

# Constants
G = 1
C = 1e20
THETA = 0.9
DT = 0.1

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def dist(p1, p2, softening=0):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + softening ** 2)

def tupleToVector2(p):
    return pygame.Vector2(p[0], p[1])