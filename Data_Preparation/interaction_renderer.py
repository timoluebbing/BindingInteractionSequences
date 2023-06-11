import pandas as pd
import pygame
import math

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir) 

from Data_Preparation.data_preparation import Preprocessor

class Interaction_Renderer():
    
    def __init__(
        self, 
        n_features,
        n_input,
        n_out, 
        interaction, 
        path=None, 
        in_tensor=None, 
        out_tensor=None
    ):
        self.n_features = n_features
        self.n_input = n_input,
        self.n_out = n_out
        self.path = path
        self.fps = 30
        self.max_fps = 201
        self.max_fps_out = 200
        self.width = 1000
        self.height = 500
        self.black, self.white, self.red = (0, 0, 0), (255, 255, 255), (255, 0, 0)
        self.blue, self.green = (0, 0, 255), (0, 255, 0)
        if path:
            self.df = pd.read_csv(path, sep=',')
        self.tensor_output = out_tensor
        self.tensor_input = in_tensor
        self.interaction = interaction
        
        self.n_dim_in  = n_input // n_features
        self.n_dim_out = n_out // n_features
    
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
        row_in = self.tensor_input[frame, :].tolist()
        row_out = self.tensor_output[frame, :].tolist()
        
        xs_in, xs_out = row_in[::self.n_dim_in], row_out[::self.n_dim_out]                      # SEHR ÜBLE magic numbers :) (abhängig von csv-format)
        xs_in = [x * self.width for x in xs_in]
        xs_out = [x * self.width for x in xs_out]
        
        ys_in, ys_out = row_in[1::self.n_dim_in], row_out[1::self.n_dim_out]
        ys_in = [y * self.height for y in ys_in]
        ys_out = [y * self.height for y in ys_out]
        
        sin_in, sin_out = row_in[2::self.n_dim_in], row_out[2::self.n_dim_out]
        cos_in, cos_out = row_in[3::self.n_dim_in], row_out[3::self.n_dim_out]
        return xs_in, ys_in, sin_in, cos_in, xs_out, ys_out, sin_out, cos_out
    
    def draw_text(self):
        my_font = pygame.font.SysFont('Arial', 50)
        legend_font = pygame.font.SysFont('Arial', 25)
        text = my_font.render(self.interaction, False, self.black)
        legend0 = legend_font.render("Input", False, self.black)
        legend1 = legend_font.render("Output", False, self.green)
        self.screen.blit(text, (30, 30))
        self.screen.blit(legend0, (self.width - 120, 20))
        self.screen.blit(legend1, (self.width - 120, 50))
    
    def draw_output(self, frame):
        self.screen.fill(self.white)
        self.draw_text()
        
        frame_mod =  frame % self.max_fps_out
        radius_in = 8
        radius_out = 10
        
        xs_in, ys_in, sin_in, cos_in, xs_out, ys_out, sin_out, cos_out = self.load_positions_at_frame_tensor(frame_mod)
        for x_in, y_in, s_in, c_in, x_out, y_out, s_out, c_out in zip(xs_in, ys_in, sin_in, cos_in, xs_out, ys_out, sin_out, cos_out):
            pygame.draw.circle(self.screen, self.black, (x_in, y_in), radius_in)
            pygame.draw.circle(self.screen, self.green, (x_out, y_out), radius_out)
            line_x_in = x_in + c_in * radius_in
            line_y_in = y_in + s_in * radius_in
            pygame.draw.line(self.screen, self.red, (x_in, y_in), (line_x_in, line_y_in), 2)
            line_x_out = x_out + c_out * radius_out
            line_y_out = y_out + s_out * radius_out
            pygame.draw.line(self.screen, self.blue, (x_out, y_out), (line_x_out, line_y_out), 2)
           
           
    def render(self, loops=1):
        
        pygame.init()
        pygame.display.set_caption(f'Sequence: {self.interaction}')
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        run = True
        frame = 0
        current_loop = 0
        
        while run:
            
            if current_loop == loops:
                break
            
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
            
            if self.tensor_input is not None and frame % self.max_fps_out == 0:
                current_loop += 1
            
            pygame.display.update()
        
        pygame.quit()
        
    def close(self):
        pygame.quit()

def main(interaction='A'):
    
    path = f"Data_Preparation/Interactions/{interaction}/interaction_{interaction}_trial_0.csv"
    print(path)

    prepro = Preprocessor(3, 6)
    renderer = Interaction_Renderer(interaction, path=path)
    # renderer.render()
    
    # df = prepro.load_preprocessed_dataframe(path)
    # df = df.iloc[:, :(3*6)] # shape like output
    # tensor_data = prepro.dataframe_to_tensor(df)
    # output_renderer = Interaction_Renderer(interaction, tensor=tensor_data)
    
    # output_renderer.render()    
    
if __name__ == "__main__":
    main('A')