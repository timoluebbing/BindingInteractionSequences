import csv
import pygame
import pymunk
import pymunk.pygame_util
import math


class Interaction():
    
    def __init__(self, width, height, fps, max_frames):
        
        pygame.init()
        self.window = pygame.display.set_mode((width, height))
        self.width, self.height = width, height
        
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.dt = 1 / self.fps
        self.max_frames = max_frames

        self.frame_id = 0
        self.recording = False
        self.data = None
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        # Add objects to the space
        self.create_boundaries()
        self.actor1, self.actor2 = self.create_actors()
        self.pressed_position = None
        self.ball = None
        self.line = None
        self.already_collided = False

        self.draw_options = pymunk.pygame_util.DrawOptions(self.window) 
        
    def create_boundaries(self):  # sourcery skip: class-extract-method
        rects = [
            # [position, size]
            [(self.width/2, self.height - 10), (self.width, 20)],   # bottom
            [(self.width/2, 10), (self.width, 20)],            # ceiling
            [(10, self.height/2), (20, self.height)],          # left wall
            [(self.width - 10, self.height/2), (20, self.height)],  # right wall
        ]    
        
        for pos, size in rects:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = pos
            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.4
            shape.friction = 0.5
            self.space.add(body, shape)
        
    def create_ball(self, radius, mass, position):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.mass = mass
        shape.elasticity = 0.9
        shape.friction = 0.4
        shape.color = (0, 255, 0, 100)
        self.space.add(body, shape)
        return shape

    def create_actors(self, interaction='A'):
        WHITE = (255, 255, 255, 100)
        rects = [
            [(200, self.height - 200), (30, 160), WHITE, 100],
            [(800, self.height - 200), (30, 160), WHITE, 200],
        ]
        actors = []
        
        for pos, size, color, mass in rects:
            body = pymunk.Body()
            body.position = pos
            shape = pymunk.Poly.create_box(body, size, radius=2)
            shape.color = color
            shape.mass = mass
            shape.elasticity = 0.4
            shape.friction = 0.4
            self.space.add(body, shape)
            actors.append(shape)
        return tuple(actors)

    def jumping_event(self, event):
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                print(self.actor2.body.body_type)
                self.actor2.body.body_type = pymunk.Body.DYNAMIC
                print(self.actor2.body)
                self.actor2.body.apply_impulse_at_local_point((0, -100000), (0,0))
    
    def apply_impulse_at_angle(self):
            self.ball.body.body_type = pymunk.Body.DYNAMIC
            angle = calculate_angle(*self.line)
            force = calculate_eucl_dis(*self.line) * 50
            fx = math.cos(angle) * force
            fy = math.sin(angle) * force
            self.ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))

    def throw_ball_event(self, event):
        
        if self.ball and self.pressed_position:
            self.line = [self.pressed_position, pygame.mouse.get_pos()]

        if not self.ball:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.pressed_position = pygame.mouse.get_pos()
                self.ball = self.create_ball(20, 10, self.pressed_position)
        elif self.pressed_position:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.apply_impulse_at_angle()
                self.pressed_position = None
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.space.remove(self.ball, self.ball.body)      
            self.ball = None
            self.already_collided = False
    
    def throw_ball_back_event(self, event):
        if not self.ball:
            return

        self.manage_collisions(0, 0)

        if self.pressed_position and self.ball:
            self.line = [self.pressed_position, pygame.mouse.get_pos()]

        # print(self.already_collided)

        if (
            self.already_collided
            and self.pressed_position
            and event.type == pygame.MOUSEBUTTONDOWN
        ):
            self.apply_impulse_at_angle()
            self.pressed_position = None
    
    def manage_collisions(self, object_a, object_b):
        self.collision_handler = self.space.add_collision_handler(object_a, object_b)
        self.collision_handler.data["surface"] = self.window
        self.collision_handler.begin = self.on_collision

    def on_collision(self, arbiter, space, data):
        if self.already_collided:
            return True
        self.already_collided = True
        print("Collision")
        if self.ball:
            self.pressed_position = self.ball.body.position
            self.space.remove(self.ball, self.ball.body)
            self.ball = self.create_ball(20, 10, self.pressed_position)

        return True

    def event_handler(self, interaction='A'):
        
        for event in pygame.event.get():
            # Close window
            if event.type == pygame.QUIT:
                # row = self.export_data_for_one_time_step()
                # print(row)
                self.is_running = False
                break

            # Start recording the sequence with keyboard press (s)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                print("Recording...")
                self.recording = True
                self.data = []

            # Interactions events:
            event_sequence = {
                'A': [self.throw_ball_event],
                'B': [self.throw_ball_event],
                'C': [self.throw_ball_event, self.throw_ball_back_event],
                'D': [self.throw_ball_event, self.jumping_event]
            }
            for func in event_sequence[interaction]:
                func(event)

    def draw(self):
        self.window.fill("white")
        
        self.space.debug_draw(self.draw_options)
        pygame.display.update()
        
    def run(self, interaction='A'):   

        while self.is_running:
            ### MAIN LOOP ###
            self.event_handler(interaction)
            self.draw()
            
            # Record the positional data for 250 frames
            if self.recording:
                if self.frame_id < self.max_frames:
                    row = self.export_data_for_one_time_step()
                    self.data.append(row)
                    self.frame_id += 1

                elif self.frame_id == self.max_frames:
                    self.recording = False
                    print("Recording finished")
                    break
                
            self.space.step(self.dt)
            self.clock.tick(self.fps)

        pygame.quit()

        return self.data
        
    def export_data_for_one_time_step(self):
        shapes = self.space.shapes[4:] # Exclude boundary shapes
        body_pos_angle_list = [
            (shape.body.position, shape.body.angle) for shape in shapes
            ]
        data = [
            [pos.x, pos.y, angle] for (pos, angle) in body_pos_angle_list
            ]
        flat_data = [x for xs in data for x in xs]
        flat_data.insert(0, self.frame_id)
        return flat_data

    def export_data_to_csv(self, header, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(self.data)

def calculate_eucl_dis(p1, p2):
        return math.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

def calculate_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def main(interaction='A'):
    
    FPS = 30
    MAX_FRAMES = 201
    WIDTH, HEIGHT = 1000, 800
    
    simulation = Interaction(WIDTH, HEIGHT, FPS, MAX_FRAMES)

    if data := simulation.run(interaction):
        
        print(f"Frames recorded: {len(data)}")

        headers = {
            'A': ["frame", "actor1_x", "actor1_y", "actor1_o", "actor2_x", "actor2_y", "actor2_o", "ball_x", "ball_y", "ball_o"],
            'B': ["frame", "actor1_x", "actor1_y", "actor1_o", "actor2_x", "actor2_y", "actor2_o", "ball_x", "ball_y", "ball_o"],
            'C': ["frame", "actor1_x", "actor1_y", "actor1_o", "actor2_x", "actor2_y", "actor2_o", "ball_x", "ball_y", "ball_o"],
            'D': ["frame", "actor1_x", "actor1_y", "actor1_o", "actor2_x", "actor2_y", "actor2_o", "ball_x", "ball_y", "ball_o"],
            # ...
            }
        header = headers[interaction]
        directory = f"Data_Preparation/Interactions/interaction_{interaction}_temp.csv"
        simulation.export_data_to_csv(header, directory)

if __name__ == "__main__":
    main('C')    