import pygame
from pygame.locals import *
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from node import *
from body import *
from utils import *

# Constants
G = 1
C = 1e20
THETA = 0.9
DT = 0.1

# Initializing pygame
pygame.init()
width, height = 800, 800
window = pygame.display.set_mode((width, height))

# Constructing a quad tree to fit the simulation
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

# Calculating gravitational forces
def calculateGravity(body, mass, pos, dt):
    radius = dist(body.pos, pos)
    if radius > body.radius:
        mag = G * mass / radius ** 2
        dir = pos - body.pos
        dir.scale_to_length(mag)
        body.vel += dir * dt
        if body.vel.magnitude() > C:
            body.vel.scale_to_length(C)

# Recursively searching quad tree to calculate gravitational forces
def handleGravity(body, root, theta, dt):
    if not root.internal and root.mass != 0:
        calculateGravity(body, root.mass, root.center, dt)
    else:
        if root.s / dist(body.pos, root.center) > theta:
            for child in root.children:
                handleGravity(body, child, theta, dt)
        else:
            calculateGravity(body, root.mass, root.center, dt)

# Recursively searching quad tree to detect merges
def handleMerges(body, root):
    global bodies
    if body.rect.colliderect(root.rect):
        if not root.internal:
            for test in root.bodies:
                if body is not test:
                    radius = dist(body.pos, test.pos)
                    if radius < body.radius + test.radius:
                        body.setVel((body.mass * body.vel + test.mass * test.vel) / (body.mass + test.mass))
                        body.setPos((body.mass * body.pos + test.mass * test.pos) / (body.mass + test.mass))
                        body.setMass(body.mass + test.mass)
                        body.setRadius(math.sqrt(body.radius ** 2 + test.radius ** 2))
                        root.bodies.remove(test)
                        bodies.remove(test)
        else:
            for child in root.children:
                handleMerges(body, child)

# Timing and events
clock = pygame.time.Clock()
fps = 100
past_pos = pygame.Vector2(0, 0)

# Boolean flags
paused = False
clicked = False
click_frame = False
gravity_on = True
merge_on = True
plot_diagnostics = False
render = False
pygame.display.set_caption("Simulation (Diagnostics {}, Render {})".format("On" if plot_diagnostics else "Off", 
                                                                           "On" if render else "Off"))

# Draw modes
trails = False
draw_tree = False
draw_gravity = False
draw_merge = False

# Setting up initial state
bodies = []

stars = [Body(400, 400, 10000, 50)]
star = stars[0]
planets = []
for i in range(4999):
    r = math.sqrt(random.random()) * (width / 2 - star.radius) + star.radius
    theta = random.random() * 2 * math.pi
    planet = Body(r * math.cos(theta) + width / 2, r * math.sin(theta) + height / 2, 0.5, 0.5)
    vel = (planet.pos - star.pos).rotate(90)
    vel.scale_to_length(math.sqrt(G * star.mass / dist(planet.pos, star.pos)) * (random.random() * 0.8 + 0.6))
    # vel.scale_to_length(random.random() * math.sqrt(G * star.mass / dist(planet.pos, star.pos)))
    # vel.scale_to_length(random.random() * 5)
    if vel.magnitude() > C:
        vel.scale_to_length(C)
    planet.setVel(vel)
    planets.append(planet)

for star in stars:
    bodies.append(star)
for planet in planets:
    bodies.append(planet)

"""
for i in range(8):
    bodies.append(Body(random.randint(100, 700), random.randint(100, 700), 25, 10))
"""

num_bodies = len(bodies)

# Diagnostics data
ticks = 0
particle_nums = []
construct_times = []
gravity_times = []
merge_times = []

# Render data
scene = []

# Game loop
sim_on = True
while sim_on:
    if not trails:
        window.fill((0, 0, 0))

    # Event handling
    for event in pygame.event.get():
        if event.type == QUIT:
            sim_on = False
        elif event.type == MOUSEBUTTONDOWN:
            event_pos = tupleToVector2(event.pos)
            clicked = True
            for body in bodies:
                if dist(body.pos, event_pos) < body.radius:
                    body.clicked = True
                    break
        elif event.type == MOUSEBUTTONUP:
            clicked = False
            for body in bodies:
                body.clicked = False
        elif event.type == MOUSEMOTION:
            event_pos = tupleToVector2(event.pos)
            body_clicked = False
            for body in bodies:
                if body.clicked:
                    body_clicked = True
                    body.setPos(body.pos + event_pos - past_pos)
            if clicked and not body_clicked:
                for body in bodies:
                    body.setPos(body.pos + event_pos - past_pos)
            past_pos = event_pos
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
            elif event.key == K_m:
                merge_on = not merge_on
            elif event.key == K_d:
                plot_diagnostics = not plot_diagnostics
                pygame.display.set_caption("Simulation (Diagnostics {}, Render {})".format("On" if plot_diagnostics else "Off", 
                                                                                           "On" if render else "Off"))
            elif event.key == K_r:
                render = not render
                pygame.display.set_caption("Simulation (Diagnostics {}, Render {})".format("On" if plot_diagnostics else "Off", 
                                                                                           "On" if render else "Off"))
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

    # Updating the frame
    if not paused or click_frame:
        start = time.perf_counter()
        root = constructQuadTree(bodies)
        construct_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if gravity_on:
            for body in bodies:
                handleGravity(body, root, THETA, DT)
        gravity_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if merge_on:
            bodies = sorted(bodies, key=lambda x: -x.radius)
            idx = 0
            while idx < len(bodies):
                handleMerges(bodies[idx], root)
                idx += 1
        merge_times.append(time.perf_counter() - start)
        particle_nums.append(len(bodies))
            
        for body in bodies:
            if body.clicked:
                body.stop()
            else:
                body.update(DT)
        click_frame = False
        ticks += 1
    else:
        root = constructQuadTree(bodies)

    # Drawing the Schwarzschild Radius of the star
    # pygame.draw.circle(window, (50, 50, 50, 0.5), star.pos, 2 * G * star.mass / C ** 2, width=5)

    # Draw modes
    if draw_tree: 
        root.draw(window)
    if draw_gravity:
        root.drawGravity(window, tupleToVector2(pygame.mouse.get_pos()), THETA)
    if draw_merge:
        body_clicked = False
        for body in bodies:
            if body.clicked:
                body_clicked = True
                root.drawMerge(window, body)
        if not body_clicked:
            pos = pygame.mouse.get_pos()
            root.drawMerge(window, Body(pos[0], pos[1], 0, 25))
            pygame.draw.circle(window, (100, 100, 100, 255), pos, 25)
    
    # Drawing the bodies
    for body in bodies:
        body.draw(window)
    pygame.display.update()
    if not paused:
        scene.append(window.copy())
    clock.tick(fps)

print("Average Construction Time: {}".format(sum(construct_times) / len(construct_times)))
print("Average Gravity Time: {}".format(sum(gravity_times) / len(gravity_times)))
print("Average Merge Time: {}".format(sum(merge_times) / len(merge_times)))

# ---------- Post simulation (in progress) ----------
if render:
    idx = 0
    left, right = False, False
    sim_on = True
    paused = True
    pygame.display.set_caption("Render")
    while sim_on:
        for event in pygame.event.get():
            if event.type == QUIT:
                sim_on = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_LEFT:
                    right = False
                    left = True
                elif event.key == K_RIGHT:
                    left = False
                    right = True
                elif event.key == K_1:
                    fps = 25
                elif event.key == K_2:
                    fps = 50
                elif event.key == K_3:
                    fps = 100
                elif event.key == K_4:
                    fps = 500
            elif event.type == KEYUP:
                if event.key == K_LEFT:
                    left = False
                elif event.key == K_RIGHT:
                    right = False
        window.blit(scene[idx], (0, 0))
        pygame.display.update()
        if paused:
            if left:
                idx = max(0, idx - 1)
            elif right:
                idx = min(len(scene) - 1, idx + 1)
        else:
            idx = min(len(scene) - 1, idx + 1)
        clock.tick(fps)

pygame.quit()

if plot_diagnostics:
    particle_nums = np.array(particle_nums)
    construct_times = np.array(construct_times)
    merge_times = np.array(merge_times)
    gravity_times = np.array(gravity_times)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set(figwidth=10, figheight=6)
    x = np.arange(0, ticks, 1)
    ax1.plot(x, construct_times * 1000, color="green", label="Construction Time")
    ax1.plot(x, gravity_times * 1000, color="blue", label="Gravity Time")
    ax1.plot(x, merge_times * 1000, color="red", label="Merge Time")
    ax1.set_yscale("log")
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Time (MS)")
    ax1.legend()

    ax2.plot(particle_nums, construct_times * 1000, color="green", label="Construction Time")
    ax2.plot(particle_nums, gravity_times * 1000, color="blue", label="Gravity Time")
    ax2.plot(particle_nums, merge_times * 1000, color="red", label="Merge Time")
    ax2.set_xlabel("Number of Bodies")
    ax2.set_ylabel("Time (MS)")
    ax2.legend()

    fig.suptitle("Simulation Diagnostics ({} Bodies)".format(num_bodies))
    fig.tight_layout()
    plt.show()
# ---------------------------------------------------
