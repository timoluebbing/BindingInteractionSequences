import pandas as pd
import pygame
import math

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir) 

from Data_Preparation.data_preparation import Preprocessor

class Interaction_Renderer():
    
    def __init__(self, interaction, path=None, tensor=None):
        self.path = path
        self.fps = 30
        self.max_fps = 201
        self.max_fps_out = 200
        self.width = 1000
        self.height = 500
        self.black, self.white, self.red = (0, 0, 0), (255, 255, 255), (255, 0, 0)
        if path:
            self.df = pd.read_csv(path, sep=',')
        self.tensor_output = tensor
        self.interaction = interaction
    
    ########################################################################
    # Render sequence from pandas dataframe csv file
    ########################################################################
    def load_positions_at_frame_csv(self, frame):
        row = self.df.iloc[frame]
        coordinates = row.values[1::]                    # remove frame
        xs = coordinates[::9]                            # SEHR ÜBLE magic numbers :) (abhängig von csv-format)
        ys = coordinates[1::9]
        os = coordinates[2::9]
        return xs, ys, os
    
    def draw(self, frame):
        self.screen.fill(self.white)

        my_font = pygame.font.SysFont('Arial', 50)
        text = my_font.render(self.interaction, False, self.black)
        self.screen.blit(text, (30, 30))
        
        frame_mod =  frame % self.max_fps
        
        xs, ys, angles = self.load_positions_at_frame_csv(frame_mod)
        for x, y, angle in zip(xs, ys, angles):
            pygame.draw.circle(self.screen, self.black, (x, y), 10)
            line_x = x + math.cos(angle) * 10
            line_y = y + math.sin(angle) * 10
            pygame.draw.line(self.screen, self.red, (x, y), (line_x, line_y), 2)        
        
    ########################################################################
    # Render predicted output sequence from torch tensor
    ########################################################################
    def load_positions_at_frame_tensor(self, frame):
        row = self.tensor_output[frame, :].tolist()
        xs = row[::6]                        # SEHR ÜBLE magic numbers :) (abhängig von csv-format)
        xs = [x * self.width for x in xs]
        ys = row[1::6]
        ys = [y * self.height for y in ys]
        sin = row[2::6]
        cos = row[3::6]
        return xs, ys, sin, cos
    
    def draw_output(self, frame):
        self.screen.fill(self.white)

        my_font = pygame.font.SysFont('Arial', 50)
        text = my_font.render(self.interaction, False, self.black)
        self.screen.blit(text, (30, 30))
        
        frame_mod =  frame % self.max_fps_out
        
        xs, ys, sin, cos = self.load_positions_at_frame_tensor(frame_mod)
        for x, y, sin, cos in zip(xs, ys, sin, cos):
            pygame.draw.circle(self.screen, self.black, (x, y), 10)
            line_x = x + cos * 10
            line_y = y + sin * 10
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
            
            if self.path is not None:
                self.draw(frame)
            else:
                self.draw_output(frame)
                
            self.clock.tick(self.fps)
            frame += 1
            
            pygame.display.update()
        
        pygame.quit()
        


def main(interaction='A'):
    
    path = f"Data_Preparation/Interactions/{interaction}/interaction_{interaction}_trial_0.csv"
    print(path)

    prepro = Preprocessor(3, 6)
    renderer = Interaction_Renderer(interaction, path=path)
    # renderer.render()
    
    df = prepro.load_preprocessed_dataframe(path)
    df = df.iloc[:, :(3*6)] # shape like output
    tensor_data = prepro.dataframe_to_tensor(df)
    output_renderer = Interaction_Renderer(interaction, tensor=tensor_data)
    
    output_renderer.render()    
    
if __name__ == "__main__":
    main('A')