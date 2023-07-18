import torch
import numpy as np
from numpy import random
from torch.nn import PairwiseDistance


class LSTM_Utils():
    
    """
        Parent class for common lstm operations
        
    """
    
    def __init__(
        self,
        batch_size, 
        hidden_num,
        teacher_forcing_steps,
        teacher_forcing_dropouts, 
        layer_norm, 
        num_dim, 
        num_feat,
        num_independent_feat,
        num_interactions,
        num_output,
    ):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 
        print(f'DEVICE TrainM: {self.device}')
        
        self.batch_size = batch_size
        self.num_dim = num_dim
        self.num_feat = num_feat
        self.num_independent_feat = num_independent_feat
        self.num_interactions = num_interactions
        self.num_output = num_output
        self.hidden_num = hidden_num
        self.layer_norm = layer_norm
        
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_dropouts = teacher_forcing_dropouts
        self.dropout_chance = 0.0
        # random.seed(0)
        self.random_thresholds = random.random_sample((teacher_forcing_steps,))
        
    
    def forward_pass(self, seq, interaction, state, outs, j):
            
        if self.teacher_forcing_dropouts and j < self.teacher_forcing_steps:
            not_a_dropout = self.random_thresholds[j] > self.dropout_chance
            self.dropout_chance += 1 / self.teacher_forcing_steps

        if (
            j < self.teacher_forcing_steps           # -> Ignore dropout functionality
                and not self.teacher_forcing_dropouts  
            or j < self.teacher_forcing_steps        # -> Use tf until tf_steps is reached
                and not_a_dropout                    # and the current time step is not a dropout
        ):
            _input = seq[j, :, :]
            output, state = self.model.forward(_input, interaction, state)
        else:
            # concat motor forces and distances to previous output
            output = self.closed_loop_input(seq, j, outs[-1])
            # Closed loop lstm forward pass without teacherforcing
            output, state = self.model.forward(output, interaction, state)
                    
        outs.append(output)
        
        return output, state, outs

    
    def reset_dropout_chance(self):
        self.dropout_chance = 0.0
        

    def closed_loop_input(self, seq, j, output):
        distances = self.calculate_new_distances(output)
        motor_column = self.num_dim*self.num_feat
        motor_force = seq[j, :, motor_column:motor_column+2].squeeze()
        return torch.cat([output, motor_force, distances], dim=1)


    def calculate_new_distances(self, out: torch.Tensor):
        # shape out: [seq_len=1, batch_size, features]
        t = out.squeeze()
        # a1, a2, b = t[:, [0,1]], t[:, [6,7]], t[:, [12,13]]
        a1, a2, b = (
            t[:, [2*i + i*(self.num_dim-2), 2*(i+1) + i*(self.num_dim-2)]] 
            for i in range(self.num_feat)
        )
        
        # abstrahieren für n_obj!=3 mit [a1, a2, b, ...] als list und:
        # combs = it.combinations(tensors, 2)
        # dann loop über combs und dist anwenden
        
        dist = PairwiseDistance(p=2)
        dis_a1_a2 = dist(a1, a2)
        dis_a1_b  = dist(a1, b)
        dis_b_a2  = dist(b, a2)
        
        return torch.stack([dis_a1_a2, dis_a1_b, dis_b_a2], dim = 1) # shape (batchsize x 3)
    