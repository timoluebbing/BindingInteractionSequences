import pandas as pd
import pygame
import math


class Interaction_Renderer():
    
    def __init__(self, path):
        self.fps = 30
        self.max_fps = 201
        self.width = 1000
        self.height = 800
        self.black, self.white, self.red = (0, 0, 0), (255, 255, 255), (255, 0, 0)
        self.df = pd.read_csv(path, sep=',')
        self.num_features = (len(self.df.columns) - 1) // 2
        self.interaction_label = path[-18:-5]
        
    def load_positions_at_frame_t(self, frame):
        row = self.df.iloc[frame]
        coordinates = row.values[1::]                    # remove frame
        xs = coordinates[::3]
        ys = coordinates[1::3]
        os = coordinates[2::3]
        return xs, ys, os
    
    def draw(self, frame):
        self.screen.fill(self.white)
        frame_mod =  frame % self.max_fps
        
        xs, ys, angles = self.load_positions_at_frame_t(frame_mod)
        for x, y, angle in zip(xs, ys, angles):
            pygame.draw.circle(self.screen, self.black, (x, y), 10)
            line_x = x + math.cos(angle) * 10
            line_y = y + math.sin(angle) * 10
            pygame.draw.line(self.screen, self.red, (x, y), (line_x, line_y), 2)
        
        
    def render(self):
        
        pygame.init()
        pygame.display.set_caption(f'Sequence: {self.interaction_label}')
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        run = True
        frame = 0
        
        while run:
            
            for event in pygame.event.get():
                # Close window
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            self.draw(frame)
            
            self.clock.tick(self.fps)
            frame += 1
            
            pygame.display.update()
        
        pygame.quit()
        


def main():
    
    path = "Data_Preparation/Interactions/interaction_A1.csv"

    renderer = Interaction_Renderer(path)
    x, y, o = renderer.load_positions_at_frame_t(20)
    print(x, y, o)

    renderer.render()
    
if __name__ == "__main__":
    main()