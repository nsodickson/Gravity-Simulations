import pygame
from pygame.locals import *
from pygame.math import Vector2
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

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
width, height = 801, 801
center = Vector2(width / 2, height / 2)
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 25)


# Calculating gravitational forces
def calculateGravity(body: Iterable[Body], mass: float, pos: Vector2, gravity_cutoff: float=2) -> None:
    r_mag = dist(pos, body.pos)
    if r_mag > gravity_cutoff * body.radius:
        r_hat = (body.pos - pos) / r_mag
        acc = (G * mass * (-1 * r_hat)) / (r_mag ** 2)
        pe = (-1 * G * body.mass * mass) / r_mag
        body.setAcc(body.acc + acc)
        body.setPE(body.pe + pe)


# Calculating merge effects
def calculateMerge(body: Body, test: Body, bodies: Iterable[Body], root: Node) -> None:
    radius = dist(body.pos, test.pos)
    if radius < body.radius + test.radius:
        body.setVel((body.mass * body.vel + test.mass * test.vel) / (body.mass + test.mass))
        body.setPos((body.mass * body.pos + test.mass * test.pos) / (body.mass + test.mass))
        body.setMass(body.mass + test.mass)
        body.setRadius(math.sqrt(body.radius ** 2 + test.radius ** 2))
        root.bodies.remove(test)
        bodies.remove(test)


# Calculation collision effects
def calculateCollision(body: Body, test: Body, elasticity: float=1) -> None:
    r_mag = dist(body.pos, test.pos)
    if r_mag < body.radius + test.radius:
        r_hat = (body.pos - test.pos) / r_mag
        theta_hat = r_hat.rotate(90)
        
        body_vel_r = r_hat * (r_hat * body.vel)
        body_vel_theta = theta_hat * (theta_hat * body.vel)
        body_acc_r = r_hat * (r_hat * body.acc)
        body_acc_theta = theta_hat * (theta_hat * body.acc)

        test_vel_r = r_hat * (r_hat * test.vel)
        test_vel_theta = theta_hat * (theta_hat * test.vel)
        test_acc_r = r_hat * (r_hat * test.acc)
        test_acc_theta = theta_hat * (theta_hat * test.acc)

        total_mass = body.mass + test.mass
        mass_dif = abs(body.mass - test.mass)
        elastic_vel = r_hat * (mass_dif * body_vel_r.magnitude() + 2 * test.mass * test_vel_r.magnitude()) / total_mass

        body.setVel(body_vel_theta + elasticity * elastic_vel)
        body.setAcc(body_acc_theta)  # Normal force cancels our r_hat acceleration


# Constructing a quad tree to fit the simulation
def _constructQuadTree(bodies: Iterable[Body]) -> Node:
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
def _handleGravity(body: Body, root: Node, theta: float, *args, **kwargs) -> None:
    if not root.internal and root.mass != 0:
        calculateGravity(body, root.mass, root.center, *args, **kwargs)
    else:
        d = dist(body.pos, root.center)
        if d == 0 or root.s / d > theta:
            for child in root.children:
                _handleGravity(body, child, theta, *args, **kwargs)
        else:
            calculateGravity(body, root.mass, root.center, *args, **kwargs)


# Recursively searching quad tree to detect merges
def _handleMerge(body: Body, bodies: Iterable[Body], root: Node, *args, **kwargs) -> None:
    if body.rect.colliderect(root.rect):
        if not root.internal:
            for test in root.bodies:
                if body is not test:
                    calculateMerge(body, test, bodies, root, *args, **kwargs)
        else:
            for child in root.children:
                _handleMerge(body, bodies, child, *args, **kwargs)


# Recursively searching quad tree to detect collisions
def _handleCollision(body: Body, root: Node, *args, **kwargs) -> None:
    if body.rect.colliderect(root.rect):
        if not root.internal:
            for test in root.bodies:
                if body is not test:
                    calculateCollision(body, test, *args, **kwargs)
        else:
            for child in root.children:
                _handleCollision(body, child, *args, **kwargs)


def constructQuadTree(bodies: Iterable[Body]) -> tuple[Node, float]:
    start = time.perf_counter()
    root = _constructQuadTree(bodies)
    end = time.perf_counter()
    return root, end - start


def handleGravity(bodies: Iterable[Body], root: Node, theta: float, *args, **kwargs) -> float:
    start = time.perf_counter()
    for body in bodies:
        body.setAcc(Vector2(0, 0))
        _handleGravity(body, root, theta, *args, **kwargs)
    end = time.perf_counter()
    return end - start


def handleMerge(bodies: Iterable[Body], root: Node, *args, **kwargs) -> float:
    start = time.perf_counter()
    idx = 0
    while idx < len(bodies):
        _handleMerge(bodies[idx], bodies, root, *args, **kwargs)
        idx += 1
    end = time.perf_counter()
    return end - start


def handleCollision(bodies: Iterable[Body], root: Node, *args, **kwargs) -> float:
    start = time.perf_counter()
    for body in bodies:
        _handleCollision(body, root, *args, **kwargs)
    end = time.perf_counter()
    return end - start


# Running simulation
def run(bodies: Iterable[Body], 
        dt: float=DT, theta: 
        float=THETA, 
        init_fps: int=100, 
        merge: bool=False, 
        collision: bool=False, 
        start_paused: bool=False, 
        vel_scale: float=1, 
        acc_scale: float=1, 
        elasticity: float=1) -> None:
    # Timing and events
    fps = init_fps
    past_pos = tupleToVector2(pygame.mouse.get_pos())

    # Boolean flags and click modes
    paused = start_paused
    clicked = False
    click_frame = False
    gravity_on = True
    merge_on = merge
    collision_on = collision
    clicked_body_idx = None

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
    collision_times = np.empty(0)

    # Game loop
    root, con_time = constructQuadTree(bodies)
    sim_on = True
    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {}, Collision: {})".format("On" if gravity_on else "Off", 
                                                                                           "On" if merge_on else "Off",
                                                                                           "On" if collision_on else "Off"))
    while sim_on:
        window.fill((0, 0, 0))

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            elif event.type == MOUSEBUTTONDOWN:
                event_pos = tupleToVector2(event.pos)
                clicked = True
                for i, body in enumerate(bodies):
                    if dist(body.pos, event_pos) < body.radius:
                        clicked_body_idx = i
                        body.stop()
                        break
            elif event.type == MOUSEBUTTONUP:
                clicked = False
                clicked_body_idx = None
            elif event.type == MOUSEMOTION:
                event_pos = tupleToVector2(event.pos)
                if clicked:
                    if clicked_body_idx is None:
                        for body in bodies:
                            body.setPos(body.pos + event_pos - past_pos)
                    else:
                        clicked_body = bodies[clicked_body_idx]
                        clicked_body.setPos(clicked_body.pos + event_pos - past_pos)
                past_pos = event_pos
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:  # Pause simulation
                    paused = not paused
                elif event.key == K_RETURN:  # Run a single frame if paused
                    if paused:
                        click_frame = True
                elif event.key == K_UP:  # Decrease theta parameter
                    theta = min(2.0, theta + 0.05)
                elif event.key == K_DOWN:  # Increase theta parameter
                    theta = max(0, theta - 0.05)
                elif event.key == K_LEFT:  # Decrease fps
                    fps = init_fps / 5
                elif event.key == K_RIGHT:  # Increase fps
                    fps = init_fps * 5
                elif event.key == K_g:  # Toggle gravity
                    gravity_on = not gravity_on
                    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {}, Collision: {})".format("On" if gravity_on else "Off", 
                                                                                                           "On" if merge_on else "Off",
                                                                                                           "On" if collision_on else "Off"))
                elif event.key == K_c:  # Enter the center of mass frame
                    for body in bodies:
                        body.setPos(body.pos + (center - root.center))
                elif event.key == K_m:  # Toggle merge
                    merge_on = not merge_on
                    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {}, Collision: {})".format("On" if gravity_on else "Off", 
                                                                                                           "On" if merge_on else "Off",
                                                                                                           "On" if collision_on else "Off"))
                elif event.key == K_k:  # Toggle collision (EXPERIMENTAL)
                    collision_on = not collision_on
                    pygame.display.set_caption("Simulation (Gravity: {}, Merge: {}, Collision: {})".format("On" if gravity_on else "Off", 
                                                                                                           "On" if merge_on else "Off",
                                                                                                           "On" if collision_on else "Off"))
                elif event.key == K_1:  # Tree draw mode
                    draw_gravity = False
                    draw_merge = False
                    draw_tree = not draw_tree
                elif event.key == K_2:  # Gravity tree draw mode
                    draw_tree = False
                    draw_merge = False
                    draw_gravity = not draw_gravity
                elif event.key == K_3:  # Merge tree draw mode
                    draw_tree = False
                    draw_gravity = False
                    draw_merge = not draw_merge
                elif event.key == K_4:  # Acceleration draw mode
                    draw_acc = not draw_acc
                elif event.key == K_5:  # Velocity draw mode
                    draw_vel = not draw_vel
                elif event.key == K_6:  # Energy draw mode
                    draw_energy = not draw_energy
                elif event.key == K_ESCAPE:  # Leave simulation
                    return
            elif event.type == KEYUP:
                if event.key == K_LEFT or event.key == K_RIGHT:
                    fps = init_fps

        # Updating the frame
        if not paused or click_frame:
            root, con_time = constructQuadTree(bodies)
            construct_times = np.append(construct_times, con_time)

            if gravity_on:
                grav_time = handleGravity(bodies, root, theta)
                gravity_times = np.append(gravity_times, grav_time)
            else:
                gravity_times = np.append(gravity_times, np.nan)

            if merge_on:
                merge_time = handleMerge(bodies, root)
                merge_times = np.append(merge_times, merge_time)
            else:
                merge_times = np.append(merge_times, np.nan)
            
            if collision_on:  # (EXPERIMENTAL)
                col_time = handleCollision(bodies, root, elasticity=elasticity)
                collision_times = np.append(collision_times, col_time)
            else:
                collision_times = np.append(collision_times, np.nan)

            body_nums = np.append(body_nums, len(bodies))

            ke = 0
            pe = 0
            p = Vector2(0, 0)
            for i, body in enumerate(bodies):
                # Acceleration dependant
                if draw_acc:
                    body.draw_acc(window, scale=acc_scale, max_length=100)
                pe += body.getPE() / 2

                if i == clicked_body_idx:
                    body.stop()
                else:
                    body.update(dt)
                
                # Velocity dependant
                if draw_vel:
                    body.draw_vel(window, scale=vel_scale, max_length=100)
                ke += body.getKE()
                p += body.getP()
            click_frame = False
        else:
            root, con_time = constructQuadTree(bodies)

            # Still calculates gravity to display accelerations
            if gravity_on:
                grav_time = handleGravity(bodies, root, theta)
            
            ke = 0
            pe = 0
            p = Vector2(0, 0)
            for body in bodies:
                # Acceleration dependant
                if draw_acc:
                    body.draw_acc(window, scale=acc_scale, max_length=100)
                pe += body.getPE() / 2
                
                body.setAcc(Vector2(0, 0))
                body.setPE(0)

                # Velocity dependant
                if draw_vel:
                    body.draw_vel(window, scale=vel_scale, max_length=100)
                ke += body.getKE()
                p += body.getP()

        # Drawing the bodies
        for body in bodies:
            body.draw(window)

        # Draw modes
        if draw_tree: 
            root.draw(window)
        if draw_gravity:
            root.drawGravity(window, tupleToVector2(pygame.mouse.get_pos()), theta)
        if draw_merge:
            if clicked_body_idx is None:
                pos = pygame.mouse.get_pos()
                root.drawMerge(window, Body(pos[0], pos[1], 0, 25))
                pygame.draw.circle(window, (100, 100, 100, 255), pos, 25)
            else:
                clicked_body = bodies[clicked_body_idx]
                root.drawMerge(window, clicked_body)
        if draw_energy:
            ke_img = font.render("Kinetic Energy: {:.2f}".format(ke), True, RED) 
            pe_img = font.render("Potential Energy: {:.2f}".format(pe), True, RED) 
            tme_img = font.render("Total Energy: {:.2f}".format(ke + pe), True, RED) 
            p_img = font.render("Momentum: {:.2f}".format(p.magnitude()), True, RED) 
            window.blit(ke_img, (5, 5))
            window.blit(pe_img, (5, 25))
            window.blit(tme_img, (5, 45))
            window.blit(p_img, (5, 65))

        pygame.display.update()
        clock.tick(fps)


# Render a simulation
def render(bodies: Iterable[Body], 
           dt: float=DT, 
           theta: float=THETA, 
           frames: int=1000, 
           init_fps: int=100, 
           merge: bool=False, 
           collision: bool=False, 
           draw_vel: bool=False, 
           draw_acc: bool=False, 
           draw_energy: bool=False,
           vel_scale: float=1, 
           acc_scale: float=1) -> None:
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
    collision_times = np.empty(0)

    # Render Game loop
    render_on = True
    pygame.display.set_caption("Rendering")
    while render_on and ticks < frames:
        screen.fill((0, 0, 0))
        window.fill((255, 255, 255))

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return

        # Updating the frame
        root, con_time = constructQuadTree(bodies)
        construct_times = np.append(construct_times, con_time)

        grav_time = handleGravity(bodies, root, theta)
        gravity_times = np.append(gravity_times, grav_time)

        if merge:
            merge_time = handleMerge(bodies, root)
            merge_times = np.append(merge_times, merge_time)
        else:
            merge_times = np.append(merge_times, np.nan)

        if collision:  # (EXPERIMENTAL)
            col_time = handleCollision(bodies, root)
            collision_times = np.append(collision_times, col_time)
        else:
            collision_times = np.append(collision_times, np.nan)

        body_nums = np.append(body_nums, len(bodies))

        ke = 0
        pe = 0
        p = Vector2(0, 0)
        for body in bodies:
            if draw_acc:
                body.draw_acc(screen, scale=acc_scale, max_length=100)
            pe += body.getPE() / 2

            body.update(dt)
            if draw_vel:
                body.draw_vel(screen, scale=vel_scale, max_length=100)
            ke += body.getKE()
            p += body.getP()
        
        for body in bodies:
            body.draw(screen)
        
        if draw_energy:
            ke_img = font.render("Kinetic Energy: {:.2f}".format(ke), True, RED) 
            pe_img = font.render("Potential Energy: {:.2f}".format(pe), True, RED) 
            tme_img = font.render("Total Energy: {:.2f}".format(ke + pe), True, RED) 
            p_img = font.render("Momentum: {:.2f}".format(p.magnitude()), True, RED) 
            screen.blit(ke_img, (5, 5))
            screen.blit(pe_img, (5, 25))
            screen.blit(tme_img, (5, 45))
            screen.blit(p_img, (5, 65))
            
        scene.append(screen.copy())

        # Drawing the rendering loading bar
        pygame.draw.rect(window, (0, 0, 0), (width / 2 - 100, height / 2 - 15, 200, 30), 2)
        pygame.draw.rect(window, (255, 0, 0), (width / 2 - 98, height / 2 - 13, (ticks / frames) * 196, 26))
        pygame.display.update()

        ticks += 1

    # Timing and events
    ticks = 0
    fps = init_fps
    past_pos = tupleToVector2(pygame.mouse.get_pos())
    offset = Vector2(0, 0)

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
                quit()
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
                elif event.key == K_c:
                    offset = Vector2(0, 0)
                elif event.key == K_ESCAPE:
                    return
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


# Trace future paths
def run_trails(bodies: Iterable[Body], 
               dt: float=DT, 
               theta: float=THETA, 
               trail_length: int=100, 
               vel_scale: float=100, 
               vel_click_size: float=5, 
               *args, 
               **kwargs) -> None:
    # Timing and events
    past_pos = tupleToVector2(pygame.mouse.get_pos())

    # Boolean flags and click modes
    paused = False
    clicked = False
    gravity_on = True
    trails_on = True
    draw_vel = False
    draw_energy = True
    clicked_body_idx = None
    clicked_vel_idx = None

    # Trails
    colors = [body.color for body in bodies]

    # Game loop
    root, con_time = constructQuadTree(bodies)
    sim_on = True
    pygame.display.set_caption("Trail Simulation (Gravity: {}, Trail Length: {})".format("On" if gravity_on else "Off",
                                                                                         trail_length))
    while sim_on:
        window.fill((0, 0, 0))

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                quit()
            elif event.type == MOUSEBUTTONDOWN:
                event_pos = tupleToVector2(event.pos)
                clicked = True
                if draw_vel:
                    for i, body in enumerate(bodies):
                        if dist(body.pos + body.vel * vel_scale, event_pos) < vel_click_size:
                            clicked_vel_idx = i
                            break
                if clicked_vel_idx is None:
                    for i, body in enumerate(bodies):
                        if dist(body.pos, event_pos) < body.radius:
                            clicked_body_idx = i
                            break
            elif event.type == MOUSEBUTTONUP:
                clicked = False
                clicked_vel_idx = None
                clicked_body_idx = None
            elif event.type == MOUSEMOTION:
                event_pos = tupleToVector2(event.pos)
                if clicked:
                    if clicked_vel_idx is not None:
                        clicked_body = bodies[clicked_vel_idx]
                        clicked_body.setVel(clicked_body.vel + (event_pos - past_pos) / vel_scale)
                    elif clicked_body_idx is not None:
                        clicked_body = bodies[clicked_body_idx]
                        clicked_body.setPos(clicked_body.pos + event_pos - past_pos)
                    else:
                        for body in bodies:
                            body.setPos(body.pos + event_pos - past_pos)
                past_pos = event_pos
            elif event.type == KEYDOWN:
                if event.key == K_UP:  # Increase theta parameter
                    theta = min(2.0, theta + 0.05)
                elif event.key == K_DOWN:  # Decrease theta parameter
                    theta = max(0, theta - 0.05)
                elif event.key == K_LEFT:  # Decrease trail length
                    trail_length = max(0, trail_length - 100)
                    pygame.display.set_caption("Trail Simulation (Gravity: {}, Trail Length: {})".format("On" if gravity_on else "Off",
                                                                                                         trail_length))
                elif event.key == K_RIGHT:  # Increase trail length
                    trail_length += 100
                    pygame.display.set_caption("Trail Simulation (Gravity: {}, Trail Length: {})".format("On" if gravity_on else "Off",
                                                                                                         trail_length))
                elif event.key == K_SPACE:  # Pause simulation
                    paused = not paused
                elif event.key == K_e:  # Energy draw mode
                    draw_energy = not draw_energy
                elif event.key == K_r:  # Run simulation
                    run(bodies, dt=dt, *args, **kwargs)
                    pygame.display.set_caption("Trail Simulation (Gravity: {}, Trail Length: {})".format("On" if gravity_on else "Off",
                                                                                                         trail_length))
                elif event.key == K_t:  # Trail draw mode
                    trails_on = not trails_on
                elif event.key == K_g:  # Toggle gravity 
                    gravity_on = not gravity_on
                    pygame.display.set_caption("Trail Simulation (Gravity: {}, Trail Length: {})".format("On" if gravity_on else "Off",
                                                                                                         trail_length))
                elif event.key == K_c:  # Enter center of mass frame
                    for body in bodies:
                        body.setPos(body.pos + (center - root.center))
                elif event.key == K_v:  # Velocity draw and adjust mode
                    if draw_vel:  # If adjusting a vel, stop that adjustment
                        clicked_vel_idx = None
                    draw_vel = not draw_vel
                elif event.key == K_ESCAPE:  # Leave simulation
                    return

        if trails_on:
            # Copying bodies and initializing trails
            trails = []
            _bodies = []
            for body in bodies: 
                _bodies.append(Body(x=body.pos.x, 
                                    y=body.pos.y, 
                                    mass=body.mass, 
                                    radius=body.radius, 
                                    init_vel_x=body.vel.x, 
                                    init_vel_y=body.vel.y))
                trails.append([])

            # Generating Trails
            for i in range(trail_length + 1):
                for event in pygame.event.get(eventtype=QUIT):
                    if event.type == QUIT:
                        quit()

                # Updating the frame
                _root, con_time = constructQuadTree(_bodies)
                if gravity_on:
                    handleGravity(_bodies, _root, theta)
                for i, _body in enumerate(_bodies):
                    trails[i].append(_body.pos.copy())
                    _body.update(dt)

            # Drawing trails
            for trail, color in zip(trails, colors):
                for i in range(trail_length):
                    pygame.draw.line(window, color, trail[i], trail[i + 1])

        # Calculating gravity to display energy
        root, con_time = constructQuadTree(bodies)
        grav_time = handleGravity(bodies, root, theta)

        # Drawing bodies
        ke = 0
        pe = 0
        p = Vector2(0, 0)
        for body in bodies:

            # Calculating gravity to display energy
            pe += body.getPE() / 2
            ke += body.getKE()
            p += body.getP()
            body.setAcc(Vector2(0, 0))
            body.setPE(0)

            body.draw(window)
            if draw_vel:
                body.draw_vel(window, scale=vel_scale, color=WHITE, max_length=10000)
                pygame.draw.circle(window, WHITE, (body.pos + body.vel * vel_scale), vel_click_size)
        
        # Displaying energy
        if draw_energy:
            ke_img = font.render("Kinetic Energy: {:.2f}".format(ke), True, RED) 
            pe_img = font.render("Potential Energy: {:.2f}".format(pe), True, RED) 
            tme_img = font.render("Total Energy: {:.2f}".format(ke + pe), True, RED) 
            p_img = font.render("Momentum: {:.2f}".format(p.magnitude()), True, RED) 
            window.blit(ke_img, (5, 5))
            window.blit(pe_img, (5, 25))
            window.blit(tme_img, (5, 45))
            window.blit(p_img, (5, 65))

        pygame.display.update()


# Setting up initial state ------------------------------------------------------------------------
bodies = []
setup = "trails".upper()

# Protoplanetary Disk
if setup == "DISK":
    bodies.append(Body(width / 2, height / 2, 100000, 50))
    main_star = bodies[0]
    for i in range(4999):
        r = math.sqrt(random.random()) * (min(width, height) / 2 - main_star.radius) + main_star.radius
        theta = random.random() * 2 * math.pi
        planet = Body(r * math.cos(theta) + width / 2, r * math.sin(theta) + height / 2, 0.5, 0.5)
        vel = (planet.pos - main_star.pos).rotate(90)
        vel.scale_to_length(math.sqrt(G * main_star.mass / dist(planet.pos, main_star.pos)) * (random.random() * 0.8 + 0.6))
        # vel.scale_to_length(random.random() * math.sqrt(G * star.mass / dist(planet.pos, star.pos)))
        # vel.scale_to_length(random.random() * 5)
        if vel.magnitude() > C:
            vel.scale_to_length(C)
        planet.setVel(vel)
        bodies.append(planet)

# Grid
elif setup == "GRID":
    star1 = Body(width // 4, 3 * height // 4, 100000, 50)
    star2 = Body(3 * width // 4, 3 * height // 4, 100000, 50)
    star3 = Body(width // 2, height // 4, 100000, 50)
    bodies = [star1, star2]
    for i in range(0, width, 20):
        for n in range(0, width, 20):
            bodies.append(Body(i, n, 0.5, 0.5))

# Small Number of Bodies
elif setup == "NBODY":
    for i in range(10):
        body = Body(random.randint(50, 750), random.randint(50, 750), 100, 10)
        body.setVel(Vector2(random.randint(-5, 5) / 10, random.randint(-5, 5) / 10))
        body.setVel(Vector2(0, 0))
        bodies.append(body)

# Trails Setup
elif setup == "TRAILS":
    body = Body(500, 600, 10000, 20, color=RED)
    bodies.append(body)

    body = Body(300, 400, 100, 12, color=GREEN)
    bodies.append(body)

    body = Body(500, 400, 100, 12, color=BLUE)
    bodies.append(body)
# -------------------------------------------------------------------------------------------------

# run(bodies=bodies, dt=0.1, theta=THETA, init_fps=100 / DT, merge=False, start_paused=True, vel_scale=10, acc_scale=100, elasticity=1)
# render(bodies=bodies, dt=0.1, theta=THETA, frames=2500, init_fps=80, merge=True, collision=False, draw_vel=False, draw_acc=False, draw_energy=True, vel_scale=1, acc_scale=1)
run_trails(bodies=bodies, dt=0.5, theta=THETA, trail_length=2500, vel_scale=10, vel_click_size=3)

pygame.quit()
