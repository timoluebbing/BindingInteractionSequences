import sys
import csv
import pygame
import pymunk
import pymunk.pygame_util
import math
import pandas as pd

    
def create_boundaries(space, width, height):
    rects = [
        # [position, size]
        [(width/2, height - 10), (width, 20)],   # bottom
        [(width/2, 10), (width, 20)],            # ceiling
        [(10, height/2), (20, height)],          # left wall
        [(width - 10, height/2), (20, height)],  # right wall
    ]    
    
    for pos, size in rects:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.4
        shape.friction = 0.5
        space.add(body, shape)
    
def create_ball(space, radius, mass, position):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.mass = mass
    shape.elasticity = 0.9
    shape.friction = 0.4
    shape.color = (0, 255, 0, 100)
    space.add(body, shape)
    return shape

def create_actors(space, width, height, interaction='A'):
    WHITE = (255, 255, 255, 100)
    rects = [
        [(200, height - 200), (30, 160), WHITE, 100],
        [(800, height - 200), (30, 160), WHITE, 100],
    ]
    
    for pos, size, color, mass in rects:
        body = pymunk.Body()
        body.position = pos
        shape = pymunk.Poly.create_box(body, size, radius=2)
        shape.color = color
        shape.mass = mass
        shape.elasticity = 0.4
        shape.friction = 0.4
        space.add(body, shape)

def calculate_eucl_dis(p1, p2):
    return math.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

def calculate_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def throw_ball_event(space, event, ball, pressed_position, line):
    
    def apply_impulse_at_angle(ball, line):
        ball.body.body_type = pymunk.Body.DYNAMIC
        angle = calculate_angle(*line)
        force = calculate_eucl_dis(*line) * 50
        fx = math.cos(angle) * force
        fy = math.sin(angle) * force
        ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))
        return None
    
    if ball and pressed_position:
        line = [pressed_position, pygame.mouse.get_pos()]

    if not ball:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed_position = pygame.mouse.get_pos()
            ball = create_ball(space, 20, 10, pressed_position)
    elif pressed_position:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed_position = apply_impulse_at_angle(ball, line)
    elif event.type == pygame.MOUSEBUTTONDOWN:
        space.remove(ball, ball.body)      
        ball = None
    return space, ball, pressed_position

def draw(space, window, draw_options):
    window.fill("white")
    
    space.debug_draw(draw_options)
    pygame.display.update()
    
def run(window, width, height, fps, max_frames, interaction='A'):
    run = True
    clock = pygame.time.Clock()
    dt = 1 / fps

    frame_id = 0
    recording = False
    data = None
    
    space = pymunk.Space()
    space.gravity = (0, 981)

    # Add objects to the space
    create_boundaries(space, width, height)
    create_actors(space, width, height)
    pressed_position = None
    ball = None
    line = None

    draw_options = pymunk.pygame_util.DrawOptions(window)    

    while run:
        for event in pygame.event.get():

            # Close window
            if event.type == pygame.QUIT:
                row = export_data_for_one_time_step(space, frame_id)
                print(row)
                run = False
                break

            # Start recording the sequence with keyboard press (s)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                print("Recording...")
                recording = True
                data = []

            # Interactions events:
            space, ball, pressed_position = throw_ball_event(
                space, event, ball, pressed_position, line)        

        draw(space, window, draw_options)
        space.step(dt)
        clock.tick(fps)
        
        # Record the positional data for 250 frames
        if recording and frame_id < max_frames:
            row = export_data_for_one_time_step(space, frame_id)
            data.append(row)
            frame_id += 1
            
        elif recording and frame_id == max_frames:
            recording = False
            print("Recording finished")
            break

    pygame.quit()
    
    return data
    
def export_data_for_one_time_step(space, frame_id):
    shapes = space.shapes[4:] # Exclude boundary shapes
    # body_list = [shape.body for shape in shapes]
    body_position_list = [shape.body.position for shape in shapes]
    body_pos_angle_list = [(shape.body.position, shape.body.angle) for shape in shapes]
    positions = [[pos.x, pos.y, angle] for (pos, angle) in body_pos_angle_list]
    flat_positions = [x for xs in positions for x in xs]
    flat_positions.insert(0, frame_id)
    return flat_positions

def export_data_to_csv(data, header, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(data)

def main(interaction='A'):
    
    pygame.init()

    FPS = 30
    MAX_FRAMES = 201
    WIDTH, HEIGHT = 1000, 800
    window = pygame.display.set_mode((WIDTH, HEIGHT))

    if data := run(window, WIDTH, HEIGHT, FPS, MAX_FRAMES):
        
        print(f"Frames recorded: {len(data)}")

        headers = {
            'A': ["frame", "actor1_x", "actor1_y", "actor1_o", "actor2_x", "actor2_y", "actor2_o", "ball_x", "ball_y", "ball_o"],
            'B': ["frame", "actor1_x", "actor1_y", "actor2_x", "actor2_y", "..."]
            # ...
            }
        header = headers[interaction]
        directory = f"Data_Preparation/Interactions/interaction_{interaction}.csv"
        export_data_to_csv(data, header, directory)

if __name__ == "__main__":
    main()    