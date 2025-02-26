import pygame
from pygame.locals import *
from pygame.math import Vector2
import math

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def dist(p1: Vector2, p2: Vector2) -> float:
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def tupleToVector2(p: tuple[float, float]) -> float:
    return pygame.Vector2(p[0], p[1])
