
import torch 
from torch import nn
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from torch.optim import Adam, AdamW
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


class LSTM_Trainer():

    """
        Class to train core LSTM model on interactions sequences.
        
    """

    def __init__(self, 
            loss_function, 
            learning_rate, 
            betas, 
            weight_decay, 
            batch_size, 
            hidden_num,
            teacher_forcing_steps, 
            layer_norm, 
            num_dim, 
            num_feat,
            independent_feat,
        ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 
        print(f'DEVICE TrainM: {self.device}')

        self.model = CORE_NET(
            input_size=num_dim*num_feat+independent_feat+4, 
            hidden_layer_size=hidden_num, 
            layer_norm=layer_norm
        )
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.teacher_forcing_steps = teacher_forcing_steps
        
        self.optimizer2 = Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            betas=betas, 
            weight_decay=weight_decay
        )
        
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )
        
        # Vielleicht in Zukunft
        # self.lr_scheduler = CyclicLR(...)

        print('Initialized model!')
        print(self.model)
        print(self.loss_function)
        print(self.optimizer)

    def train(self, 
              epochs, 
              dataloader,         
              save_path, 
        ):

        losses = []

        for ep in range(epochs):

            single_losses = torch.tensor([0.0], device=self.device)

            self.model.zero_grad()
            self.optimizer.zero_grad()

            for seq, label, interaction in tqdm(dataloader):
                
                seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
                interaction = interaction.to(torch.int64)
                seq = seq.permute(1,0,2)
                label = label.permute(1,0,2)
                
                _, single_losses = self.train_single_sequence(seq, 
                                                              label, 
                                                              interaction, 
                                                              single_losses)

            with torch.no_grad():
                ep_loss = single_losses.clone().item()
                avg_loss = ep_loss / (len(dataloader) * self.batch_size)
                print(f'Epoch: {ep:1} - Avg. Loss: {avg_loss:10.8f} - Epoch Loss: {ep_loss:8.4f}')

                # save loss of epoch
                losses.append(ep_loss)

        self.save_model(save_path)

        return losses

    def train_single_sequence(self, 
                              seq, 
                              label, 
                              interaction, 
                              single_losses
        ):
        seq_len, batch_size, _ = seq.size()

        state = self.model.init_hidden(batch_size=batch_size)
        outs = []
        
        for j in range(seq_len):
            
            if j < self.teacher_forcing_steps:
                _input = seq[j, :, :]
                output, state = self.model.forward(_input, interaction, state)
            else:
                # concat motor forces and distances to previous output
                distances = self.calculate_new_distances(output)
                motor_force = seq[j, :, 18:20].squeeze() # if j < seq_len-1 else seq[0,:,18:20]
                output = torch.cat([output, motor_force, distances], dim=1)
                
                # Closed loop lstm forward pass without teacherforcing
                output, state = self.model.forward(output, interaction, state)
                
                
            outs.append(output)

        outs = torch.stack(outs)

        single_loss = self.loss_function(outs, label)
        single_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            single_losses += single_loss

        return outs, single_losses    

    def calculate_new_distances(self, out: torch.Tensor):
        # shape out: [seq_len=1, batch_size, features]
        t = out.squeeze()
        a1, a2, b = t[:, [0,1]], t[:, [6,7]], t[:, [12,13]]
        
        dist = PairwiseDistance(p=2)
        dis_a1_a2 = dist(a1, a2)
        dis_a1_b  = dist(a1, b)
        dis_b_a2  = dist(b, a2)
        
        return torch.stack([dis_a1_a2, dis_a1_b, dis_b_a2], dim = 1) # shape (batchsize x 3)
        
    
    def plot_losses(self, losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss')
        axes.set_yscale('log')
        axes.set_title('History of MSELoss during training')
        
        plt.savefig(f'{plot_path}_losses.png')
        #plt.savefig(f'{plot_path}_losses.pdf')
        plt.show()


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model was saved in: {path}')
    
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
                
            return loss
        
                
    def evaluate_model_with_renderer(self, dataloader, model_save_path, n_samples=4):
        
        example = next(iter(dataloader))
        seq, label, interaction = example
        seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
        interaction = interaction.to(torch.int64)
        
        seq = seq.permute(1,0,2)
        label = label.permute(1,0,2)
        seq_len, batch_size, num_features = seq.size()
        
        model = self.model
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        state = model.init_hidden(batch_size=batch_size)
        outs = []
        
        for j in range(seq_len):
            _input = seq[j, :, :].to(self.device)
            # print(f"input shape step {j}: {_input.shape}")
            out, state = model.forward(input_seq=_input, interaction_label=interaction, state=state)
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