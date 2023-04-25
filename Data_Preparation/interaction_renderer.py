import pandas as pd
import pygame
import math


class Interaction_Renderer():
    
    def __init__(self, path, interaction):
        self.fps = 30
        self.max_fps = 201
        self.width = 1000
        self.height = 800
        self.black, self.white, self.red = (0, 0, 0), (255, 255, 255), (255, 0, 0)
        self.df = pd.read_csv(path, sep=',')
        self.interaction = interaction
        
    def load_positions_at_frame_t(self, frame):
        row = self.df.iloc[frame]
        coordinates = row.values[1::]                    # remove frame
        xs = coordinates[::3]
        ys = coordinates[1::3]
        os = coordinates[2::3]
        return xs, ys, os
    
    def draw(self, frame):
        self.screen.fill(self.white)

        my_font = pygame.font.SysFont('Arial', 50)
        text = my_font.render(self.interaction, False, self.black)
        self.screen.blit(text, (30, 30))
        
        frame_mod =  frame % self.max_fps
        
        xs, ys, angles = self.load_positions_at_frame_t(frame_mod)
        for x, y, angle in zip(xs, ys, angles):
            pygame.draw.circle(self.screen, self.black, (x, y), 10)
            line_x = x + math.cos(angle) * 10
            line_y = y + math.sin(angle) * 10
            pygame.draw.line(self.screen, self.red, (x, y), (line_x, line_y), 2)
           
    def render(self):
        
        pygame.init()
        pygame.display.set_caption(f'Sequence: {self.interaction}')
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
        


def main(interaction='A'):
    
    path = f"Data_Preparation/Interactions/interaction_{interaction}.csv"
    path2 = f"Data_Preparation/Interactions/C/interaction_{interaction}_trial_1.csv"

    renderer = Interaction_Renderer(path2, interaction)

    renderer.render()
    
if __name__ == "__main__":
    main('C')