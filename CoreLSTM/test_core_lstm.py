
import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from torch.utils.data import DataLoader, random_split

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
        num_dim, 
        num_feat,
        num_independent_feat,
        num_interactions,
        num_output,
        model_save_path
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if num_output == 12:
            self.num_dim = int(num_output / num_feat)
            self.num_feature_types = self.num_dim // 2

        elif num_output == 18:
            self.num_dim = num_dim
            self.num_feature_types = self.num_dim // 2
            
        self.num_obj = num_feat
        self.num_interactions = num_interactions
        self.num_output = num_output

        self.batch_size = batch_size
        self.loss_function = loss_function
        self.model_save_path = model_save_path
        
        self.input_size = num_dim*num_feat

        self.model = CORE_NET(
            input_size=self.input_size+num_independent_feat+num_interactions, 
            hidden_layer_size=hidden_num, 
            output_size=num_output,
            layer_norm=layer_norm
        )

        self.load_model()
        print(f"Model load path: {model_save_path}")
        print(f'DEVICE TestM:    {self.device}')
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

    def evaluate(self, dataloader):
        
        loss = torch.tensor([0.0], device=self.device)
        
        with torch.no_grad():
            for seq, label, interaction in tqdm(dataloader):
                    
                seq, label, interaction = self.model.restructure_data(seq, label, interaction)

                seq_len, batch_size, num_features = seq.size()
        
                state = self.model.init_hidden(batch_size=batch_size)
                outs = []
                
                for j in range(seq_len):
                    _input = seq[j, :, :].to(self.device)
                    out, state = self.model.forward(input_seq=_input, interaction_label=interaction, state=state)
                    outs.append(out)
                    
                outs = torch.stack(outs).to(self.device)
                
                single_loss = self.loss_function(outs, label)
                loss += single_loss
                
            total_loss = loss.clone().item()
            avg_loss = total_loss / (len(dataloader) * self.batch_size)
            print(f'\nEvaluate: Avg. batch loss: {avg_loss:10.8f} - Total loss: {total_loss:8.4f}\n')
                
            return total_loss
    
    def evaluate_detailed(self, dataloader):
        
        batch_losses = torch.tensor([0.0], device=self.device)
        batch_losses_each_step = np.zeros(200)
        batch_losses_each_step_objects = np.zeros((self.num_obj, 200))
        batch_losses_each_step_data = np.zeros((self.num_feature_types, 200))
                            
        with torch.no_grad():
            for seq, label, interaction in tqdm(dataloader):
                    
                seq, label, interaction = self.model.restructure_data(seq, label, interaction)

                seq_len, batch_size, num_features = seq.size()
        
                state = self.model.init_hidden(batch_size=batch_size)
                outs = []
                batch_loss_each_step = np.zeros(200)
                batch_loss_each_step_objects = np.zeros((self.num_obj, 200))
                batch_loss_each_step_data = np.zeros((self.num_feature_types, 200))
                
                for j in range(seq_len):
                    _input = seq[j, :, :].to(self.device)
                    out, state = self.model.forward(input_seq=_input, interaction_label=interaction, state=state)
                    outs.append(out)
                    
                    single_loss = self.loss_function(out, label[j, :, :])
                    single_loss = single_loss.item()
                    batch_loss_each_step[j] = single_loss
                    
                    for i in range(self.num_obj):
                        # Object specific loss
                        single_object_out, single_object_label = self.get_data_by_object(out, label, i, j)
                        object_loss = self.loss_function(single_object_out, single_object_label)
                        batch_loss_each_step_objects[i, j] = object_loss
                    
                    for i in range(self.num_feature_types):
                        # Data specific loss (coords, orientation, force)
                        single_data_out, single_data_label = self.get_data_by_type(out, label, i, j)
                        data_loss = self.loss_function(single_data_out, single_data_label)
                        batch_loss_each_step_data[i, j] = data_loss
                    
                    
                outs_stacked = torch.stack(outs).to(self.device)
                batch_loss = self.loss_function(outs_stacked, label)
                
                batch_losses += batch_loss
                batch_losses_each_step += batch_loss_each_step
                batch_losses_each_step_objects += batch_loss_each_step_objects
                batch_losses_each_step_data += batch_loss_each_step_data
            
            total_loss = batch_losses.item()
            losses_each_step = batch_losses_each_step
            losses_each_step_objects = batch_losses_each_step_objects
            losses_each_step_data = batch_losses_each_step_data
        
        return total_loss, losses_each_step, losses_each_step_objects, losses_each_step_data
    
    def get_data_by_object(self, out, label, i, j):
        object_out = out[:, i*self.num_dim : (i+1)*self.num_dim]
        object_label = label[j, :, i*self.num_dim : (i+1)*self.num_dim]
        return object_out, object_label
    
    def get_data_by_type(self, out, label, i, j):
        os, ls = [], []
        
        for k in range(self.num_feature_types):
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
        
        for j in range(seq_len):
            _input = seq[j, :, :].to(self.device)
            # print(f"input shape step {j}: {_input.shape}")
            out, state = self.model.forward(input_seq=_input, interaction_label=interaction, state=state)
            outs.append(out)
            
        outs = torch.stack(outs).to(self.device)
        
        for i in range(n_samples):
            seq_index = i
            output_sequence = outs[:, seq_index, :]
            input_sequence = seq[:, seq_index, :]
            int_label = str(interaction[seq_index].item())
            
            renderer = Interaction_Renderer(
                n_features=self.num_obj,
                n_input=self.input_size,
                n_out=self.num_output,
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
        
        plt.savefig(f'{plot_path}_losses.png')
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
        
        plt.savefig(f'{plot_path}_type_losses.png')
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
        
        plt.savefig(f'{plot_path}_type_losses.png')
        plt.show()


def main(render=False):
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    
    ##### Dataset and DataLoader #####
    batch_size = 180
    seed = 0
    no_forces = True
    n_out = 12 if no_forces else 18
    
    dataset = TimeSeriesDataset(
        interaction_paths, 
        no_forces=no_forces,
        n_out=n_out,
        use_distances_and_motor=True)
    generator = torch.Generator().manual_seed(seed)
    split = [0.7, 0.15, 0.15]
    split = [0.6, 0.3, 0.1]
    
    
    _, _, test_dataset = random_split(dataset, split, generator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    ##### Model parameters #####
    hidden_num = 360
    layer_norm = True
    
    n_dim = 4 if no_forces else 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    # model_name = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_400_lnorm_tfs200'
    current_best = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200'
    current_best_dropout = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200_tfd'
    current_best_dropout_wd = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0.01_180_2000_lnorm_tfs200_tfd'
    no_forces_model = 'core_lstm_4_3_5_360_MSELoss()_0.0001_0.01_180_2000_lnorm_tfs200'

    model_name = no_forces_model
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    
    tester = LSTM_Tester(
        loss_function=criterion,
        batch_size=batch_size,
        hidden_num=hidden_num,
        layer_norm=layer_norm,
        num_dim=n_dim,
        num_feat=n_features,
        num_independent_feat=n_independent,
        num_interactions=n_interactions,
        model_save_path=model_save_path
    )

    print("\nTest dataset:\n")
    print(f"Model in evaluation mode: {not tester.model.training}")
    _ = tester.evaluate(test_dataloader)
    total_loss, losses, obj_losses, type_losses = tester.evaluate_detailed(test_dataloader)
    print(f"\n Evaluation: Total loss of {total_loss:4f} - Sum step losses of {sum(losses)}")
    
    print(total_loss)
    print(sum(losses))
    print(np.sum(obj_losses, axis=None))
    print(np.sum(type_losses, axis=None))
    
    test_loss_path = f"CoreLSTM/testing_predictions/test_loss/{model_name}"
    tester.plot_losses_steps(losses, test_loss_path)
    tester.plot_losses_objects(obj_losses, test_loss_path)
    tester.plot_losses_types(type_losses, test_loss_path)

    if render:
        # Check prediction for one example with renderer
        tester.evaluate_model_with_renderer(
            # train_dataloader, 
            test_dataloader,
            n_samples=5
        )
    
if __name__ == '__main__':
    main()