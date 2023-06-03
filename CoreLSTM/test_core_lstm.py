
import torch 
from torch import nn
import matplotlib.pyplot as plt
from numpy import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from torch.optim import Adam, AdamW
from torch.nn import PairwiseDistance

from tqdm import tqdm
import sys
import copy
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

    def __init__(self, 
            loss_function, 
            batch_size, 
            hidden_num,
            layer_norm, 
            num_dim, 
            num_feat,
            independent_feat,
            model_save_path
        ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.model_save_path = model_save_path

        self.model = CORE_NET(
            input_size=num_dim*num_feat+independent_feat+4, 
            hidden_layer_size=hidden_num, 
            layer_norm=layer_norm
        )
        self.load_model()
        
        print(f'DEVICE TrainM: {self.device}')
        print(f'Model: {self.model}')
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

    def evaluate(self, dataloader):
        
        loss = torch.tensor([0.0], device=self.device)
        self.model.eval()
        
        with torch.no_grad():
            for seq, label, interaction in tqdm(dataloader):
                    
                seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
                interaction = interaction.to(torch.int64)
                seq = seq.permute(1,0,2)
                label = label.permute(1,0,2)

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
                
            ep_loss = loss.clone().item()
            avg_loss = ep_loss / (len(dataloader) * self.batch_size)
            print(f'\nEvaluate: Avg. batch loss: {avg_loss:10.8f} - Total loss: {ep_loss:8.4f}\n')
                
            return ep_loss
        
                
    def evaluate_model_with_renderer(self, dataloader, n_samples=4):
        
        seq, label, interaction = next(iter(dataloader))
        
        seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
        interaction = interaction.to(torch.int64)
        seq = seq.permute(1,0,2)
        label = label.permute(1,0,2)
        
        seq_len, batch_size, num_features = seq.size()
        
        # model = self.model
        # model.load_state_dict(torch.load(model_save_path))
        # model.eval()
        #### -> das ist jetzt in der init drin
        
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
            
            renderer = Interaction_Renderer(int_label, in_tensor=input_sequence, out_tensor=output_sequence)
            renderer.render(loops=1)
            renderer.close()
    
    def plot_losses(self, losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss (log scaled)')
        axes.set_yscale('log')
        axes.set_title('History of MSELoss during training')
        
        plt.savefig(f'{plot_path}_losses.png')
        #plt.savefig(f'{plot_path}_losses.pdf')
        plt.show()
