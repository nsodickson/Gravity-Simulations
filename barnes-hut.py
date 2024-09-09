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

# Initializing pygame
pygame.init()
width, height = 800, 800
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 25)


# Calculating gravitational forces
def calculateGravity(body, mass, pos):
    radius = dist(body.pos, pos)
    if radius > body.radius:
        mag = G * mass / radius ** 2
        acc = pos - body.pos
        acc.scale_to_length(mag)
        body.setAcc(body.acc + acc)
        body.setPE(body.pe + -G * body.mass * mass / radius)


# Constructing a quad tree to fit the simulation
def _constructQuadTree(bodies):
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


# Recursively searching quad tree to calculate gravitational forces
def _handleGravity(body, root, theta):
    if not root.internal and root.mass != 0:
        calculateGravity(body, root.mass, root.center)
    else:
        if root.s / dist(body.pos, root.center) > theta:
            for child in root.children:
                _handleGravity(body, child, theta)
        else:
            calculateGravity(body, root.mass, root.center)


# Recursively searching quad tree to detect merges
def _handleMerge(body, bodies, root):
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
                _handleMerge(body, bodies, child)


def constructQuadTree(*args, **kwargs):
    start = time.perf_counter()
    root = _constructQuadTree(*args, **kwargs)
    end = time.perf_counter()
    return root, end - start


def handleGravity(bodies, *args, **kwargs):
    start = time.perf_counter()
    for body in bodies:
        _handleGravity(body, *args, **kwargs)
    end = time.perf_counter()
    return end - start


def handleMerge(bodies, *args, **kwargs):
    start = time.perf_counter()
    idx = 0
    while idx < len(bodies):
        _handleMerge(bodies[idx], bodies, *args, **kwargs)
        idx += 1
    end = time.perf_counter()
    return end - start


def run(bodies, init_fps=100, merge=False):
    global THETA, DT

    # Timing and events
    ticks = 0
    fps = init_fps
    past_pos = pygame.Vector2(0, 0)

    # Boolean flags
    paused = False
    clicked = False
    click_frame = False
    gravity_on = True
    merge_on = merge

    # Draw modes
    draw_tree = False
    draw_gravity = False
    draw_merge = False
    draw_acc = False
    draw_vel = False
    draw_energy = True

    # Diagnostics data
    body_nums = np.empty(0)
    construct_times = np.empty(0)
    gravity_times = np.empty(0)
    merge_times = np.empty(0)
    ke = 0
    pe = 0
    p = 0

    # Game loop
    sim_on = True
    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {})".format("On" if gravity_on else "Off", 
                                                                            "On" if merge_on else "Off"))
    while sim_on:
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
                    fps = init_fps / 5
                elif event.key == K_RIGHT:
                    fps = init_fps * 5
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
                    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {})".format("On" if gravity_on else "Off", 
                                                                                            "On" if merge_on else "Off"))
                elif event.key == K_m:
                    merge_on = not merge_on
                    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {})".format("On" if gravity_on else "Off", 
                                                                                            "On" if merge_on else "Off"))
                elif event.key == K_1:
                    draw_gravity = False
                    draw_merge = False
                    draw_tree = not draw_tree
                elif event.key == K_2:
                    draw_tree = False
                    draw_merge = False
                    draw_gravity = not draw_gravity
                elif event.key == K_3:
                    draw_tree = False
                    draw_gravity = False
                    draw_merge = not draw_merge
                elif event.key == K_4:
                    draw_acc = not draw_acc
                elif event.key == K_5:
                    draw_vel = not draw_vel
                elif event.key == K_6:
                    draw_energy = not draw_energy
            elif event.type == KEYUP:
                if event.key == K_LEFT or event.key == K_RIGHT:
                    fps = init_fps

        # Updating the frame
        if not paused or click_frame:
            root, con_time = constructQuadTree(bodies)
            construct_times = np.append(construct_times, con_time)

            if gravity_on:
                grav_time = handleGravity(bodies, root, THETA)
                gravity_times = np.append(gravity_times, grav_time)
            else:
                gravity_times = np.append(gravity_times, np.nan)

            if merge_on:
                merge_time = handleMerge(bodies, root)
                merge_times = np.append(merge_times, merge_time)
            else:
                merge_times = np.append(merge_times, np.nan)

            body_nums = np.append(body_nums, len(bodies))

            ke = 0
            pe = 0
            p = 0  
            for body in bodies:
                if body.clicked:
                    body.stop()
                else:
                    body.update(DT)
                
                # Updating energy
                ke += body.getKE()
                pe += body.getPE() / 2
                p += body.getP()

                # Drawing acceleration and velocity
                if draw_acc:
                    body.draw_acc(window, scale=10, max_length=100)
                if draw_vel:
                    body.draw_vel(window, scale=1, max_length=100)
                
                body.setAcc(pygame.Vector2(0, 0))
                body.setPE(0)

            click_frame = False

            ticks += 1
        else:
            root = constructQuadTree(bodies)

            if gravity_on:
                grav_time = handleGravity(bodies, root, THETA)
                gravity_times = np.append(gravity_times, grav_time)
            else:
                gravity_times = np.append(gravity_times, np.nan)

        # Drawing the bodies
        for body in bodies:
            body.draw(window)

        # Draw modes (2)
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
        if draw_energy:
            ke_img = font.render("Kinetic Energy: {:.2f}".format(ke), True, RED) 
            pe_img = font.render("Potential Energy: {:.2f}".format(pe), True, RED) 
            tme_img = font.render("Total Energy: {:.2f}".format(ke + pe), True, RED) 
            p_img = font.render("Momentum: {:.2f}".format(p), True, RED) 
            window.blit(ke_img, (5, 5))
            window.blit(pe_img, (5, 25))
            window.blit(tme_img, (5, 45))
            window.blit(p_img, (5, 65))

        pygame.display.update()
        clock.tick(fps)

    print("Average Construction Time: {}".format(np.nanmean(construct_times) if ~np.isnan(construct_times).any() else np.nan))
    print("Average Gravity Time: {}".format(np.nanmean(gravity_times) if ~np.isnan(gravity_times).any() else np.nan))
    print("Average Merge Time: {}".format(np.nanmean(merge_times) if ~np.isnan(merge_times).any() else np.nan))

    pygame.quit()


def render(bodies, frames=1000, init_fps=100, merge=False):
    global THETA, DT

    # Render data
    scene = []
    screen = window.copy()

    # Timing and events
    ticks = 0

    # Diagnostics data
    body_nums = np.empty(0)
    construct_times = np.empty(0)
    gravity_times = np.empty(0)
    merge_times = np.empty(0)

    # Game loop
    render_on = True
    pygame.display.set_caption("Rendering")
    while render_on and ticks < frames:
        screen.fill((0, 0, 0))
        window.fill((255, 255, 255))

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                render_on = False

        # Updating the frame
        root, con_time = constructQuadTree(bodies)
        construct_times = np.append(construct_times, con_time)

        grav_time = handleGravity(bodies, root, THETA)
        gravity_times = np.append(gravity_times, grav_time)

        if merge:
            merge_time = handleMerge(bodies, root)
            merge_times = np.append(merge_times, merge_time)
        else:
            merge_times = np.append(merge_times, np.nan)
        body_nums = np.append(body_nums, len(bodies))

        for body in bodies:
            body.update(DT, reset=True)
        
        # Drawing the bodies
        for body in bodies:
            body.draw(screen)
        scene.append(screen.copy())

        # Drawing the rendering loading bar
        pygame.draw.rect(window, (0, 0, 0), (width / 2 - 100, height / 2 - 15, 200, 30), 2)
        pygame.draw.rect(window, (255, 0, 0), (width / 2 - 98, height / 2 - 13, (ticks / frames) * 196, 26))
        pygame.display.update()

        ticks += 1
    
    print("Average Construction Time: {}".format(np.nanmean(construct_times) if ~np.isnan(construct_times).any() else np.nan))
    print("Average Gravity Time: {}".format(np.nanmean(gravity_times) if ~np.isnan(gravity_times).any() else np.nan))
    print("Average Merge Time: {}".format(np.nanmean(merge_times) if ~np.isnan(merge_times).any() else np.nan))

    # Timing and events
    ticks = 0
    fps = init_fps
    past_pos = tupleToVector2(pygame.mouse.get_pos())
    offset = pygame.Vector2(0, 0)

    # Boolean Flags
    left, right = False, False
    sim_on = True
    paused = True
    clicked = False

    idx = 0
    pygame.display.set_caption("Render {}".format("(Paused)" if paused else ""))
    while sim_on:
        window.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == QUIT:
                sim_on = False
            elif event.type == MOUSEBUTTONDOWN:
                clicked = True
            elif event.type == MOUSEBUTTONUP:
                clicked = False
            elif event.type == MOUSEMOTION:
                event_pos = tupleToVector2(event.pos)
                if clicked:
                    offset += event_pos - past_pos
                past_pos = event_pos
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                    pygame.display.set_caption("Render {}".format("(Paused)" if paused else ""))
                elif event.key == K_LEFT:
                    right = False
                    left = True
                elif event.key == K_RIGHT:
                    left = False
                    right = True
                elif event.key == K_1:
                    fps = init_fps
                elif event.key == K_2:
                    fps = init_fps / 5
                elif event.key == K_3:
                    fps = init_fps / 2
                elif event.key == K_4:
                    fps = init_fps * 2
                elif event.key == K_5:
                    fps = init_fps * 5
            elif event.type == KEYUP:
                if event.key == K_LEFT:
                    left = False
                elif event.key == K_RIGHT:
                    right = False
        window.blit(scene[idx], offset)
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


# Setting up initial state ------------------------------------------------------------------------
bodies = []

# """
stars = [Body(400, 400, 100000, 50)]
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
# """

"""
for i in range(3):
    body = Body(random.randint(50, 750), random.randint(50, 750), 100, 10)
    body.setVel(pygame.Vector2(random.randint(-5, 5) / 10, random.randint(-5, 5) / 10))
    bodies.append(body)
"""
# -------------------------------------------------------------------------------------------------

run(bodies, init_fps=100 / DT)
# render(bodies, frames=10000, init_fps = 100 / DT, merge=False)
