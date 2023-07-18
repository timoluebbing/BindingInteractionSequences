
import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from torch.utils.data import DataLoader
from torch.nn import PairwiseDistance

from tqdm import tqdm
import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET
from Data_Preparation.data_preparation import Preprocessor
from Data_Preparation.interaction_dataset import TimeSeriesDataset
from Data_Preparation.interaction_renderer import Interaction_Renderer


class LSTM_Tester():

    """
        Class to train core LSTM model on interactions sequences.
        
    """

    def __init__(
        self, 
        loss_function, 
        batch_size, 
        hidden_num,
        layer_norm, 
        timesteps,
        teacher_forcing_steps,
        teacher_forcing_dropouts,
        num_dim, 
        num_feat,
        num_independent_feat,
        num_interactions,
        num_output,
        model_save_path,
        random_labels=False,
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if num_output == 12:
            self.num_dim = int(num_output / num_feat)
            self.num_feature_types = self.num_dim // 2

        elif num_output == 18:
            self.num_dim = num_dim
            self.num_feature_types = self.num_dim // 2
        
        self.timesteps = timesteps - 1 # input und label um einen step versetzt
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_dropouts = teacher_forcing_dropouts
        self.dropout_chance = 0.0
        random.seed(0)
        self.random_thresholds = random.random_sample((self.teacher_forcing_steps,))
        
        self.num_obj = num_feat
        self.num_interactions = num_interactions
        self.num_output = num_output

        self.batch_size = batch_size
        self.loss_function = loss_function
        self.model_save_path = model_save_path
        self.random_labels = random_labels
        
        self.input_size = num_dim*num_feat

        self.model = CORE_NET(
            input_size=self.input_size+num_independent_feat+num_interactions, 
            batch_size=batch_size,
            hidden_layer_size=hidden_num, 
            output_size=num_output,
            layer_norm=layer_norm,
            random_labels=self.random_labels,
        )

        self.load_model()
        print(f"Model load path: {model_save_path}")
        print(f'DEVICE TestM:    {self.device}')
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()


    def reset_dropout_chance(self):
        self.dropout_chance = 0.0


    def forward_pass1(self, seq, interaction, state, outs, j):
            
        _input = seq[j, :, :].to(self.device)
        out, state = self.model.forward(input_seq=_input, interaction_label=interaction, state=state)
        outs.append(out)
        
        return out, state, outs
    
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
        
        
    def closed_loop_input(self, seq, j, output):
        distances = self.calculate_new_distances(output)
        motor_column = self.num_dim*self.num_obj
        motor_force = seq[j, :, motor_column:motor_column+2].squeeze()
        return torch.cat([output, motor_force, distances], dim=1)


    def calculate_new_distances(self, out: torch.Tensor):
        # shape out: [seq_len=1, batch_size, features]
        t = out.squeeze()
        # a1, a2, b = t[:, [0,1]], t[:, [6,7]], t[:, [12,13]]
        a1, a2, b = (
            t[:, [2*i + i*(self.num_dim-2), 2*(i+1) + i*(self.num_dim-2)]] 
            for i in range(self.num_obj)
        )
        
        # abstrahieren für n_obj!=3 mit [a1, a2, b, ...] als list und:
        # combs = it.combinations(tensors, 2)
        # dann loop über combs und dist anwenden
        
        dist = PairwiseDistance(p=2)
        dis_a1_a2 = dist(a1, a2)
        dis_a1_b  = dist(a1, b)
        dis_b_a2  = dist(b, a2)
        
        return torch.stack([dis_a1_a2, dis_a1_b, dis_b_a2], dim = 1) # shape (batchsize x 3)
    
        
    def evaluate(self, dataloader):
        
        loss = torch.tensor([0.0], device=self.device)
        
        with torch.no_grad():
            for seq, label, interaction in tqdm(dataloader):
                    
                seq, label, interaction = self.model.restructure_data(seq, label, interaction)

                seq_len, batch_size, num_features = seq.size()
        
                state = self.model.init_hidden(batch_size=batch_size)
                outs = []
                self.reset_dropout_chance()

                for j in range(seq_len):
                    _, state, outs = self.forward_pass(seq, interaction, state, outs, j)
                    
                outs = torch.stack(outs).to(self.device)
                
                single_loss = self.loss_function(outs, label)
                loss += single_loss
                
            total_loss = loss.clone().item()
            avg_loss = total_loss / (len(dataloader) * self.batch_size)
            print(f'\nEvaluate: Avg. batch loss: {avg_loss:10.8f} - Total loss: {total_loss:8.4f}\n')
                
            return total_loss
        

    def evaluate_detailed(self, dataloader):
        
        batch_losses = torch.tensor([0.0], device=self.device)
        batch_losses_each_step = np.zeros(self.timesteps)
        batch_losses_each_step_objects = np.zeros((self.num_obj, self.timesteps))
        batch_losses_each_step_data = np.zeros((self.num_feature_types, self.timesteps))
                            
        with torch.no_grad():
            for seq, label, interaction in tqdm(dataloader):   
                seq, label, interaction = self.model.restructure_data(seq, label, interaction)
                seq_len, batch_size, num_features = seq.size()
                state = self.model.init_hidden(batch_size=batch_size)
                outs = []
                self.reset_dropout_chance()
                
                batch_loss_each_step = np.zeros(self.timesteps)
                batch_loss_each_step_objects = np.zeros((self.num_obj, self.timesteps))
                batch_loss_each_step_data = np.zeros((self.num_feature_types, self.timesteps))
                
                for j in range(seq_len):
                    out, state, outs = self.forward_pass(seq, interaction, state, outs, j)
                    
                    (batch_loss_each_step, 
                     batch_loss_each_step_objects, 
                     batch_loss_each_step_data) = self.losses_single_obj_type(
                                                    j, 
                                                    label, 
                                                    batch_loss_each_step, 
                                                    batch_loss_each_step_objects, 
                                                    batch_loss_each_step_data, 
                                                    out)
                     
                outs_stacked = torch.stack(outs).to(self.device)
                batch_loss = self.loss_function(outs_stacked, label)
                
                batch_losses += batch_loss
                batch_losses_each_step += batch_loss_each_step
                batch_losses_each_step_objects += batch_loss_each_step_objects
                batch_losses_each_step_data += batch_loss_each_step_data
            
            total_loss = batch_losses.item()
        
        return total_loss, batch_losses_each_step, batch_losses_each_step_objects, batch_losses_each_step_data


    def losses_single_obj_type(
        self, 
        j, 
        label, 
        batch_loss_each_step, 
        batch_loss_each_step_objects, 
        batch_loss_each_step_data, 
        out
    ):  
        # Single time step loss
        single_loss = self.loss_function(out, label[j, :, :])
        single_loss = single_loss.item()
        batch_loss_each_step[j] = single_loss
                    
        for i in range(self.num_obj):
            # Object specific loss
            single_object_out, single_object_label = self.get_data_by_object(out, label, i, j)
            object_loss = self.loss_function(single_object_out, single_object_label)
            batch_loss_each_step_objects[i, j] = object_loss
                    
        for i in range(self.num_feature_types):
            # Data specific loss (coords, orientation, [force])
            single_data_out, single_data_label = self.get_data_by_type(out, label, i, j)
            data_loss = self.loss_function(single_data_out, single_data_label)
            batch_loss_each_step_data[i, j] = data_loss
            
        return batch_loss_each_step, batch_loss_each_step_objects, batch_loss_each_step_data
    
    
    def get_data_by_object(self, out, label, i, j):
        object_out = out[:, i*self.num_dim : (i+1)*self.num_dim]
        object_label = label[j, :, i*self.num_dim : (i+1)*self.num_dim]
        return object_out, object_label
    
    
    def get_data_by_type(self, out, label, i, j):
        os, ls = [], []
        
        for k in range(self.num_obj):
            o = out[:, 2*i + (self.num_dim*k) : 2*(i+1) + (self.num_dim*k)]
            os.append(o)
            l = label[j, :, 2*i + (self.num_dim*k) : 2*(i+1) + (self.num_dim*k)]
            ls.append(l)
        
        type_out = torch.cat(os, dim=1)
        type_label = torch.cat(ls, dim=1)
        
        return type_out, type_label
                
                
    def evaluate_model_with_renderer(self, dataloader, n_samples=4):
        
        seq, label, interaction = next(iter(dataloader))
        seq, label, interaction = self.model.restructure_data(seq, label, interaction)
        
        seq_len, batch_size, num_features = seq.size()
        
        state = self.model.init_hidden(batch_size=batch_size)
        outs = []
        self.reset_dropout_chance()
        
        for j in range(seq_len):
            _, state, outs = self.forward_pass(seq, interaction, state, outs, j)

        outs = torch.stack(outs).to(self.device)
        
        self.render_prediction(n_samples, seq, interaction, outs)


    def render_prediction(self, n_samples, seq, interaction, outs):
        for i in range(n_samples):
            seq_index = i
            output_sequence = outs[:, seq_index, :]
            input_sequence = seq[:, seq_index, :]
            int_label = str(interaction[seq_index].item())
            
            renderer = Interaction_Renderer(
                n_features=self.num_obj,
                n_input=self.input_size,
                n_out=self.num_output,
                timesteps=self.timesteps + 1,
                interaction=int_label, 
                in_tensor=input_sequence, 
                out_tensor=output_sequence)
            renderer.render(loops=1)
            renderer.close()
    
    
    def plot_losses_steps(self, losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.12, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('sequence time steps')
        axes.set_ylabel('loss')
        # axes.set_yscale('log')
        axes.set_title('MSELoss for each test prediction time step')
        
        plt.savefig(f'{plot_path}_losses.png', dpi=300)
        plt.show()
    
    
    def plot_losses_objects(self, object_losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.12, 0.1, 0.8, 0.8]) 
        for i in range(object_losses.shape[0]):
            axes.plot(object_losses[i]) 
        axes.legend(['Actor 1', 'Actor 2', 'Ball'])
        axes.grid(True)
        axes.set_xlabel('sequence time steps')
        axes.set_ylabel('loss (object specific)')
        # axes.set_yscale('log')
        axes.set_title('MSELoss for each test prediction time step')
        
        plt.savefig(f'{plot_path}_obj_losses.png', dpi=300)
        plt.show()
    
        
    def plot_losses_types(self, type_losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.12, 0.1, 0.8, 0.8]) 
        for i in range(type_losses.shape[0]):
            axes.plot(type_losses[i]) 
        axes.legend(['Coordinates', 'Orientation', 'Impact force'])
        axes.grid(True)
        axes.set_xlabel('sequence time steps')
        axes.set_ylabel('loss (type specific)')
        # axes.set_yscale('log')
        axes.set_title('MSELoss for each test prediction time step')
        
        plt.savefig(f'{plot_path}_type_losses.png', dpi=300)
        plt.show()

