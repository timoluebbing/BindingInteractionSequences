import csv
import pygame
import pymunk
import pymunk.pygame_util
import math
import random
import numpy as np

FPS = 30
MAX_FRAMES = 201
WIDTH, HEIGHT = 1000, 500
NUM_TRIALS = 300

class Interaction():
    
    def __init__(self, width, height, fps, max_frames, num_trials=300, interaction='A'):
        
        pygame.init()
        self.window = pygame.display.set_mode((width, height))
        self.width, self.height = width, height
        
        self.is_running = True
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.dt = 1 / self.fps
        self.max_frames = max_frames

        self.interaction = interaction
        self.num_trials = num_trials
        self.current_trial = 0
        self.next_trial = self.current_trial + 1
        self.frame_id = 0
        self.recording = False
        self.data = None
        
        # Pymunk Space and physics
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        # Add objects to the space
        self.create_boundaries()
        
        # Actors
        actor_weights_by_interaction = {'A': 260, 'B':80, 'C':270, 'D':200}
        self.actor_weight = actor_weights_by_interaction[self.interaction]
        self.init_random_actor_positions(num_sequences=num_trials, seed=None)
        self.actor1, self.actor2 = None, None # self.create_actors()
        self.add_actors()
        
        # Ball
        self.ball = None
        self.add_ball_at_random_actor()
        self.throw_time = np.random.randint(15, 40)
        print(self.throw_time)
        
        # Extra functionality
        self.pressed_position = None
        self.line = None
        self.already_collided = False
        self.already_jumped = False
        self.already_thrown = False
        self.collision_detected_at_frame = 0

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
    
    def add_ball(self, position):
        self.ball = self.create_ball(radius=15, mass=10, position=position)
        
    def add_ball_at_random_actor(self):
        actor1_x = self.actor1_positions[self.current_trial]
        actor2_x = self.actor2_positions[self.current_trial]

        ball_spawn_choice = np.random.uniform(0, 1)
        ball_spawn_height = np.random.uniform(0.75, 0.9)

        ball_x = None
        ball_y = self.height - self.actor_height * ball_spawn_height
        if ball_spawn_choice < 0.5: 
            ball_x = actor1_x + self.actor_width + 8
            self.ball_spawned_at_actor = self.actor1
        else: 
            ball_x = actor2_x - self.actor_width - 8
            self.ball_spawned_at_actor = self.actor2

        self.add_ball(position=(ball_x, ball_y))       
        
    def remove_ball(self):
        self.space.remove(self.ball, self.ball.body)      
        self.ball = None

    def create_actors(self, pos_a, pos_b):
        WHITE = (255, 255, 255, 100)
        self.actor_width = 25
        self.actor_height = 100
        rects = [                            # +10 to account for boundaries
            [(pos_a, self.height - self.actor_height+10), (self.actor_width, self.actor_height), WHITE, self.actor_weight],
            [(pos_b, self.height - self.actor_height+10), (self.actor_width, self.actor_height), WHITE, self.actor_weight],
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
    
    def init_random_actor_positions(self, num_sequences, seed=None):
        if seed:
            np.random.seed(seed)
        self.actor1_positions = np.random.uniform(low=110, high=390, size=num_sequences)
        self.actor2_positions = np.random.uniform(low=610, high=890, size=num_sequences)
    
    def get_actor_positions(self):
        return self.actor1_positions[self.current_trial], self.actor2_positions[self.current_trial]
    
    def add_actors(self):
        x_actor1, x_actor2 = self.get_actor_positions()
        self.actor1, self.actor2 = self.create_actors(x_actor1, x_actor2)
        
    def remove_actors(self):
        self.space.remove(self.actor1, self.actor1.body)
        self.space.remove(self.actor2, self.actor2.body)
        self.actor1, self.actor2 = None, None
    
    ######################################################################################
    # Event functionality
    ######################################################################################
    
    # Manually 
    def apply_impulse_at_angle(self):
        self.ball.body.body_type = pymunk.Body.DYNAMIC
        angle = calculate_angle(*self.line)
        force = calculate_eucl_dis(*self.line) * 50
        fx = math.cos(angle) * force
        fy = math.sin(angle) * force
        self.ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))
    
    # Automated throw
    def apply_impulse_at_random_angle(self, ball_position, actor_position, throw_back=False):
        # Custom 'random' angle for throwing in both directions + backthrowing
        ranges = {'A': (-0.2, 0.28), 'B': (0.1, 0.28), 'C': (-0.05, 0.15), 'D': (-0.2, 0.2)}
        rng = ranges[self.interaction]
        
        if (
            throw_back
            and self.ball_spawned_at_actor is self.actor1
            or not throw_back
            and self.ball_spawned_at_actor is self.actor2
        ):
            random_param = np.random.uniform( rng[0], rng[1])
            angle = 7*math.pi / 6 + random_param
        else:
            random_param = np.random.uniform( -rng[1], -rng[0])
            angle = 11*math.pi / 6 + random_param
            
        force_scale = 100
        distance = calculate_eucl_dis(ball_position, actor_position)
        force = math.sqrt(10 * distance) * force_scale

        fx = math.cos(angle) * force
        fy = math.sin(angle) * force

        self.ball.body.body_type = pymunk.Body.DYNAMIC
        self.ball.body.apply_impulse_at_local_point((fx, fy), (0, 0))
        
    def jumping_event_manually(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            print(self.actor2.body.body_type)
            self.actor2.body.body_type = pymunk.Body.DYNAMIC
            print(self.actor2.body)
            self.actor2.body.apply_impulse_at_local_point((0, -100000), (0,0))

    # Automated
    def jumping_event(self):
        
        def get_positions_and_jump(actor_with_ball, opposite_actor):
            ball_position = self.ball.body.position
            actor1_position = actor_with_ball.body.position
            actor2_position = opposite_actor.body.position
            distance_ball_actor = calculate_eucl_dis(ball_position, actor2_position)
            distance_actor_actor = calculate_eucl_dis(actor1_position, actor2_position)
            
            if math.isclose(distance_ball_actor, distance_actor_actor / 2, abs_tol=12) and not self.already_jumped:
                opposite_actor.body.body_type = pymunk.Body.DYNAMIC
                opposite_actor.body.apply_impulse_at_local_point((0, -100000), (0,0))
                self.already_jumped = True
        
        if self.already_jumped:
            return
        
        if self.ball_spawned_at_actor is self.actor1:
            get_positions_and_jump(self.actor1, self.actor2)
        else:
            get_positions_and_jump(self.actor2, self.actor1)
    
    def throw_ball_event_manually(self, event):
        
        if self.ball and self.pressed_position:
            self.line = [self.pressed_position, pygame.mouse.get_pos()]

        if not self.ball:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.pressed_position = pygame.mouse.get_pos()
                self.ball = self.create_ball(15, 10, self.pressed_position)
        elif self.pressed_position:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.apply_impulse_at_angle()
                self.pressed_position = None
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.remove_ball()
            self.already_collided = False
    
    # Automated 
    def throw_ball_event(self):
        # print(self.frame_id)
        if self.frame_id == self.throw_time:
            ball_position = self.ball.body.position
            if self.ball_spawned_at_actor is self.actor1:
                opposite_actor_position = self.actor2.body.position
            else:
                opposite_actor_position = self.actor1.body.position
            self.apply_impulse_at_random_angle(ball_position, opposite_actor_position)         
    
    def throw_ball_back_event_manually(self, event):
        if not self.ball:
            return

        self.manage_collisions(0, 0, self.on_collision_manually)

        if self.pressed_position and self.ball:
            self.line = [self.pressed_position, pygame.mouse.get_pos()]

        if (
            self.already_collided
            and self.pressed_position
            and event.type == pygame.MOUSEBUTTONDOWN
        ):
            self.apply_impulse_at_angle()
            self.pressed_position = None
    
    def on_collision_manually(self, arbiter, space, data):
        if self.already_collided:
            return True
        
        self.already_collided = True
        print("Collision")
            
        if self.ball:
            self.pressed_position = self.ball.body.position
            self.remove_ball()
            self.add_ball(self.pressed_position)

        return True
    
    # Collision handler for both manually and automated
    def manage_collisions(self, object_a, object_b, on_collision_func):
        self.collision_handler = self.space.add_collision_handler(object_a, object_b)
        self.collision_handler.data["surface"] = self.window
        self.collision_handler.begin = on_collision_func
    
    # Automated        
    def throw_ball_back_event(self):
        if not self.ball:
            return
        
        self.manage_collisions(0, 0, self.on_collision)
        # print(self.collision_detected_at_frame, self.throw_time, self.frame_id)
        if (
            self.already_collided 
            and self.collision_detected_at_frame + self.throw_time == self.frame_id  # wait random time before throwing
        ):
            print('Throwing back')
            ball_position = self.ball.body.position
            if self.ball_spawned_at_actor is self.actor1:
                opposite_actor_position = self.actor1.body.position
            else:
                opposite_actor_position = self.actor2.body.position
            self.apply_impulse_at_random_angle(ball_position, opposite_actor_position, throw_back=True)
    
    def on_collision(self, arbiter, space, data):
        if self.already_collided:
            return True
        
        if not self.ball:
            return True
        
        if self.frame_id > 25:
            print("Collision")
            self.collision_detected_at_frame = self.frame_id
            self.already_collided = True
         
        last_ball_position = self.ball.body.position
        self.remove_ball()
        self.add_ball(last_ball_position)
            
        return True
    
    def print_arbiter(self, arbiter):
        if arbiter.shapes[1] not in [self.actor1, self.actor2]:
            return
        print(f"Arbiter shapes: {arbiter.shapes}")
        print(arbiter.total_impulse, arbiter.total_ke)
    
    ######################################################################################
    # Main loop fuctionality
    ######################################################################################
    def event_handler(self, automated=True):
                
        for event in pygame.event.get():
            # Close window
            if event.type == pygame.QUIT:
                # row = self.export_data_for_one_time_step()
                # print(row)
                self.is_running = False
                break

            # Start recording the trial with keyboard press (s)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s and self.recording == False:
                print(f"Recording trial {self.current_trial}...")
                self.recording = True
                self.data = []
            
            # interaction events with keyboard presses    
            if not automated:
                event_sequence = {
                'A': [self.throw_ball_event_manually],
                'B': [self.throw_ball_event_manually],
                'C': [self.throw_ball_event_manually, self.throw_ball_back_event_manually],
                'D': [self.throw_ball_event_manually, self.jumping_event_manually]
                }
                
                for func in event_sequence[self.interaction]:
                    func(event)
                    
        # Automated interaction events:
        if automated:
            event_sequence = {
            'A': [self.throw_ball_event],
            'B': [self.throw_ball_event],
            'C': [self.throw_ball_event, self.throw_ball_back_event],
            'D': [self.throw_ball_event, self.jumping_event]
            }
             
            for func in event_sequence[self.interaction]:
                func()
                
    def draw(self):
        self.window.fill("white")
        
        self.space.debug_draw(self.draw_options)
        pygame.display.update()
        
    def run(self, automated=True):   
        
        while self.is_running:
            
            ### MAIN LOOP ###
            if self.current_trial == self.num_trials:
                print("All trials are finished and recorded :)")
                break
            
            self.event_handler(automated)
            self.draw()
            
            if self.recording:
                self.record()
            if self.current_trial == self.next_trial:
                self.transition_to_next_trial()
            
            # if self.ball:
            #     self.ball.body.each_arbiter(self.print_arbiter)
            
            self.space.step(self.dt)
            self.clock.tick(self.fps)

        pygame.quit()
        
    ######################################################################################
    # Recording functionality
    ######################################################################################
    def transition_to_next_trial(self):
        self.remove_actors()
        if self.ball:
            self.remove_ball()
        self.add_actors()
        self.add_ball_at_random_actor()
        self.next_trial += 1
        self.already_jumped = False
        self.already_collided = False
        self.already_thrown = False
        self.throw_time = np.random.randint(15, 40)
        
        # Next trial starts without key press
        print(f"Recording trial {self.current_trial}...")
        self.recording = True
        self.data = []
            
    def process_finished_recording(self):
        
        print("Recording finished")
        print(f"Frames recorded: {len(self.data)}")

        self.export_data_to_csv()

        self.frame_id = 0
        self.current_trial += 1
        self.recording = False 
        
    def record(self):
        if self.current_trial == self.next_trial:
            return
        
        # Record the positional data for 250 frames
        if self.frame_id < self.max_frames:
            if self.frame_id % 25 == 0:
                print(f"Frame {self.frame_id}")
                
            row = self.export_data_for_one_time_step()
            self.data.append(row)
            self.frame_id += 1

        elif self.frame_id == self.max_frames:
            self.process_finished_recording()  
            
    ######################################################################################
    # Functionality for extracting data from each simulation
    ######################################################################################
    def get_arbiter_data(self, arbiter):
        if arbiter.shapes[1] not in [self.actor1, self.actor2, self.ball]:
            self.ti, self.ke = None, None
            return
        self.ti, self.ke = arbiter.total_impulse, arbiter.total_ke
    
    def get_collision_forces(self, shape):
        try:
            shape.body.each_arbiter(self.get_arbiter_data)
            if self.ke is not None and self.ke > 0:
                print("recording collision forces")
                total_impulse_x, total_impulse_y = self.ti
                print(shape, total_impulse_x, total_impulse_y, self.ke)
                return total_impulse_x, total_impulse_y, self.ke
        except Exception:
            return 0.0, 0.0, 0.0
    
    def export_data_for_one_time_step(self):
        shapes = self.space.shapes[4:] # Exclude boundary shapes
        body_data_list = [
            (shape.body.position, 
             shape.body.angle,
             self.get_collision_forces(shape),
             shape.body.velocity,
             shape.body.mass) for shape in shapes
            ]
        data = [
            [pos.x, 
             pos.y, 
             angle,
             forces[0] if forces is not None else 0.0,
             forces[1] if forces is not None else 0.0,
             forces[2] if forces is not None else 0.0,
             velocity.x,
             velocity.y,
             mass] for (pos, angle, forces, velocity, mass) in body_data_list
            ]
        flat_data = [x for xs in data for x in xs]
        flat_data.insert(0, self.frame_id)
        return flat_data

    def export_data_to_csv(self): 
        head = ["frame", "actor1_x", "actor1_y", "actor1_o", "actor1_col_force_x", "actor1_col_force_y", "actor1_ke", "actor1_vel_x", "actor1_vel_y", "actor1_mass",
                         "actor2_x", "actor2_y", "actor2_o", "actor2_col_force_x", "actor2_col_force_y", "actor2_ke", "actor2_vel_x", "actor2_vel_y", "actor2_mass",
                         "ball_x", "ball_y", "ball_o", "ball_col_force_x", "ball_col_force_y", "ball_ke", "ball_vel_x", "ball_vel_y", "ball_mass"]       
        headers = {
            'A': head,
            'B': head,
            'C': head,
            'D': head,
            # ... maybe diffenent with different objects
            }
        header = headers[self.interaction]
        filename = f"Data_Preparation/Interactions/{self.interaction}/interaction_{self.interaction}_trial_{self.current_trial}.csv"
        
        print(f"Exporting data to:\n {filename}")
        
        with open(filename, 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(self.data)

def calculate_eucl_dis(p1, p2):
        return math.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

def calculate_angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def main(interaction='A'):
    
    simulation = Interaction(WIDTH, HEIGHT, FPS, MAX_FRAMES, NUM_TRIALS, interaction)

    simulation.run(automated=True)

if __name__ == "__main__":
    main('C') 