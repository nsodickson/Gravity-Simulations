import pygame
from pygame.locals import *
import math
import random
import time
import sys

sys.setrecursionlimit(10000)

pygame.init()

def dist(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def tupleToVector2(p):
    return pygame.Vector2(p[0], p[1])

def insert(arr, val):
    if len(arr) == 0:
        arr.insert(0, val)
    else:
        idx = 0
        while arr[idx] > val and idx <= len(arr):
            idx += 1
        arr.insert(idx, val)


class Body(pygame.sprite.Sprite):
    def __init__(self, x, y, mass, radius):
        super(Body, self).__init__()
        
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)

        self.radius = radius
        self.mass = mass

        self.image = pygame.Surface((radius * 2, radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, (255, 0, 0), (radius, radius), radius)
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        
        self.clicked = False
    
    def setPos(self, pos):
        self.pos = pos
        self.rect.update(pos.x - self.radius, pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def setRadius(self, radius):
        self.radius = radius
        self.image = pygame.Surface((radius * 2, radius * 2), flags=SRCALPHA)
        self.image.fill((255, 255, 255, 0))
        pygame.draw.circle(self.image, (255, 0, 0), (radius, radius), radius)
        self.rect = pygame.Rect(self.pos.x - radius, self.pos.y - radius, radius * 2, radius * 2)
    
    def setMass(self, mass):
        self.mass = mass
    
    def update(self):
        self.pos += self.vel
        self.rect.update(self.pos.x - self.radius, self.pos.y - self.radius, self.radius * 2, self.radius * 2)
    
    def stop(self):
        self.vel = pygame.Vector2(0, 0)
    
    def draw(self, window):
        window.blit(self.image, (self.rect.x, self.rect.y))


class Node:
    def __init__(self, x, y, s, depth=5000):
        self.bodies = []
        self.children = []
        self.s = s
        self.rect = pygame.Rect(x, y, s, s)

        self.mass = 0
        self.weighted_pos = pygame.Vector2(0, 0)
        self.center = pygame.Vector2(x + s / 2, y + s / 2)

        self.internal = False
        self.depth = depth
    
    def addBody(self, body):
        # insert(self.bodies, body)
        self.bodies.append(body)
        self.mass += body.mass
        self.weighted_pos.x += body.pos.x * body.mass
        self.weighted_pos.y += body.pos.y * body.mass
        self.center = self.weighted_pos / self.mass

        if self.depth > 0:
            if len(self.bodies) == 2:  # The node becomes registered as internal
                self.internal = True
                self.children = [Node(self.rect.x, self.rect.y, self.s / 2, self.depth - 1),
                                 Node(self.rect.x + self.s / 2, self.rect.y, self.s / 2, self.depth - 1),
                                 Node(self.rect.x, self.rect.y + self.s / 2, self.s / 2, self.depth - 1),
                                 Node(self.rect.x + self.s / 2, self.rect.y + self.s / 2, self.s / 2, self.depth - 1)]
                for body in self.bodies:
                    idx = self.selectChild(body)
                    self.children[idx].addBody(body)
            elif len(self.bodies) > 2:
                idx = self.selectChild(body)
                self.children[idx].addBody(body)

    def selectChild(self, body):
        if body.pos.x >= self.rect.x + self.s / 2:
            if body.pos.y >= self.rect.y + self.s / 2:
                return 3
            else:
                return 1
        else:
            if body.pos.y >= self.rect.y + self.s / 2:
                return 2
            else:
                return 0
    
    def draw(self, window):
        pygame.draw.rect(window, (0, 0, 255), self.rect, 1)
        for child in self.children:
            child.draw(window)
    
    def drawGravity(self, window, pos):
        if not self.internal:
            pygame.draw.rect(window, (0, 0, 255), self.rect, 1)
        elif self.s / dist(self.center, pos) > THETA:
            for child in self.children:
                child.drawGravity(window, pos)
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

def constructQuadTree(bodies):
    min_x, max_x = 0, width
    min_y, max_y = 0, height
    for body in bodies:
        if body.pos.x < min_x:
            min_x = body.pos.x
        elif body.pos.x > max_x:
            max_x = body.pos.x
        if body.pos.y < min_y:
            min_y = body.pos.y
        elif body.pos.y > max_y:
            max_y = body.pos.y
    root = Node(min_x, min_y, max(max_x - min_x, max_y - min_y))
    for body in bodies:
        root.addBody(body)
    return root

def calculateGravity(body, mass, pos):
    radius = dist(body.pos, pos)
    if radius > body.radius:
        mag = G * body.mass * mass / radius ** 2
        dir = pos - body.pos
        dir.scale_to_length(mag / body.mass)
        body.vel += dir
        if body.vel.magnitude() > C:
            body.vel.scale_to_length(C)

def handleGravity(body, root, theta):
    if not root.internal:
        calculateGravity(body, root.mass, root.center)
    else:
        if root.s / dist(body.pos, root.center) > theta:
            for child in root.children:
                handleGravity(body, child, theta)
        else:
            calculateGravity(body, root.mass, root.center)

def handleMerges(body, root):
    global bodies
    pass

# Window and Clock Setup
width, height = 800, 800
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
fps = 100  
past_pos = pygame.Vector2(0, 0)

# Flags
paused = False
clicked = False
click_frame = False
gravity_on = True
trails = False
draw_tree = False
draw_gravity = False
draw_merge = False

# Constants
G = 1
C = 1e20
THETA = 0.5

bodies = pygame.sprite.Group()

star = Body(400, 400, 10000, 50)
planets = []
for i in range(999):
    r = math.sqrt(random.random()) * (width / 2 - star.radius) + star.radius
    theta = random.random() * 2 * math.pi
    planet = Body(r * math.cos(theta) + width / 2, r * math.sin(theta) + width / 2, 1, 1)
    vel = (planet.pos - star.pos).rotate(90)
    vel.scale_to_length(math.sqrt(G * star.mass / dist(planet.pos, star.pos)) + random.randint(-2, 2))
    # vel.scale_to_length(random.random() * math.sqrt(G * star.mass / dist(planet.pos, star.pos)))
    # vel.scale_to_length(random.random() * 5)
    if vel.magnitude() > C:
        vel.scale_to_length(C)
    planet.vel = vel
    planets.append(planet)

bodies.add(star, *planets)

"""
bodies.add(Body(random.randint(100, 700), random.randint(100, 700), 25, 10),
           Body(random.randint(100, 700), random.randint(100, 700), 25, 10),
           Body(random.randint(100, 700), random.randint(100, 700), 25, 10),
           Body(random.randint(100, 700), random.randint(100, 700), 25, 10))
"""

ticks = 0
particle_nums = []
construct_times = []
gravity_times = []

root = constructQuadTree(bodies)
game_on = True
while game_on:
    if not trails:
        window.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == QUIT:
            game_on = False
        elif event.type == MOUSEBUTTONDOWN:
            clicked = True
            for body in bodies:
                if dist(body.pos, tupleToVector2(event.pos)) < body.radius:
                    body.clicked = True
                    break
        elif event.type == MOUSEBUTTONUP:
            clicked = False
            for body in bodies:
                body.clicked = False
        elif event.type == MOUSEMOTION:
            body_clicked = False
            for body in bodies:
                if body.clicked:
                    body_clicked = True
                    body.setPos(body.pos + tupleToVector2(event.pos) - past_pos)
            if clicked and not body_clicked:
                for body in bodies:
                    body.setPos(body.pos + tupleToVector2(event.pos) - past_pos)
            past_pos = tupleToVector2(event.pos)
        elif event.type == KEYDOWN:
            if event.key == K_LEFT:
                fps = 25
            elif event.key == K_RIGHT:
                fps = 500
            elif event.key == K_UP:
                THETA = min(2.0, THETA + 0.05)
            elif event.key == K_DOWN:
                THETA = max(0, THETA - 0.05)
            elif event.key == K_RETURN:
                if paused:
                    click_frame = True
            elif event.key == K_SPACE:
                paused = not paused
            elif event.key == K_g:
                gravity_on = not gravity_on
            elif event.key == K_1:
                draw_tree = False
                draw_gravity = False
                draw_merge = False
                trails = not trails
            elif event.key == K_2:
                trails = False
                draw_gravity = False
                draw_merge = False
                draw_tree = not draw_tree
            elif event.key == K_3:
                trails = False
                draw_tree = False
                draw_merge = False
                draw_gravity = not draw_gravity
            elif event.key == K_4:
                trails = False
                draw_tree = False
                draw_gravity = False
                draw_merge = not draw_merge
        elif event.type == KEYUP:
            if event.key == K_LEFT or event.key == K_RIGHT:
                fps = 100

    if not paused or click_frame:
        start = time.perf_counter()
        root = constructQuadTree(bodies)
        construct_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if gravity_on:
            for body in bodies:
                handleGravity(body, root, THETA)
        gravity_times.append(time.perf_counter() - start)
            
        for body in bodies:
            if body.clicked:
                body.stop()
            else:
                body.update()
        click_frame = False
        ticks += 1

    # root = constructQuadTree(bodies)

    # Drawing the Schwarzschild Radius of the star
    # pygame.draw.circle(window, (50, 50, 50, 0.5), star.pos, 2 * G * star.mass / C ** 2, width=5)

    if draw_tree: 
        root.draw(window)
    if draw_gravity:
        root.drawGravity(window, tupleToVector2(pygame.mouse.get_pos()))
    if draw_merge:
        body_clicked = False
        for body in bodies:
            if body.clicked:
                body_clicked = True
                root.drawMerge(window, body)
        if not body_clicked:
            pos = pygame.mouse.get_pos()
            root.drawMerge(window, Body(pos[0], pos[1], 0, 25))
            pygame.draw.circle(window, (100, 100, 100, 0.5), pos, 25)
    bodies.draw(window)
    pygame.display.update()
    clock.tick(fps)  

    # For comparison to naive algorithm
    """
    if ticks > 100:
        game_on = False
    """

print("Average Construction Time: {}".format(sum(construct_times) / len(construct_times)))
print("Average Gravity Time: {}".format(sum(gravity_times) / len(gravity_times)))