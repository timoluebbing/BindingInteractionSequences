"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch 
from torch import nn
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

from tqdm import tqdm
import sys
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET
from Data_Preparation.data_preparation import Preprocessor
from Data_Preparation.interaction_dataset import TimeSeriesDataset


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
            layer_norm, 
            num_dim, 
            num_feat,
            independent_feat,
        ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 
        print(f'DEVICE TrainM: {self.device}')

        self._model = CORE_NET(
            input_size=num_dim*num_feat+independent_feat+4, 
            hidden_layer_size=hidden_num, 
            layer_norm=layer_norm
        )
        self.batch_size = batch_size
        self._loss_function = loss_function
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=learning_rate, 
            betas=betas, 
            weight_decay=weight_decay
        )

        print('Initialized model!')
        print(self._model)
        print(self._loss_function)
        print(self._optimizer)


    def train(self, 
              epochs, 
              dataloader,         
              save_path, 
              teacher_forcing=True, 
        ):

        losses = []

        for ep in range(epochs):

            single_losses = torch.tensor([0.0], device=self.device) 

            self._model.zero_grad()
            self._optimizer.zero_grad()

            # the following can be paralized!!!! But takes up memory!!!
            for seq, label, interaction in tqdm(dataloader):
                
                seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
                interaction = interaction.to(torch.int64)
                seq = seq.permute(1,0,2)
                label = label.permute(1,0,2)
                seq_len, batch_size, num_features = seq.size()

                state = self._model.init_hidden(batch_size=batch_size)

                outs = []

                for j in range(seq_len):
                    _input = seq[j, :, :].to(self.device)
                    out, state = self._model.forward(_input, interaction, state)
                    outs.append(out)

                outs = torch.stack(outs).to(self.device)
                single_loss = self._loss_function(outs, label)

                single_loss.backward()
                with torch.no_grad():
                    single_losses += single_loss

            self._optimizer.step()

            with torch.no_grad():
                ep_loss = single_losses.clone().detach() 
                avg_loss = ep_loss / len(dataloader)

                # save loss of epoch
                losses.append(avg_loss.item())
                # if ep%25 == 1:
                print(f'epoch: {ep:3} loss: {avg_loss.item():10.8f}')

        # print(f'epoch: {ep:3} loss: {single_losses.item():10.10f}')

        self.save_model(save_path)

        return losses
    
            #---------------------
            # for i in range(len(train_sequences)):
            #     idx = torch.randperm(num_batches)[:round((num_batches_new))]

            #     state = self._model.init_hidden(num_batches_new)

            #     ins_t = seq_ins[i][idx]          # shape: num batches x batch size x num input
            #     tars_t = seq_tars[i][idx]        # shape: num batches x batch size x num input

            #     outs = []
            #     for j in range(batch_size):
            #         _input = ins_t[:,j,:].view(num_batches_new, num_input).to(self.device)
            #         out, state = self._model.forward(_input, state)
            #         outs.append(out)

            #     outs = torch.stack(outs).to(self.device)        # shape: batch size x num batches x num input
            #     single_loss = self._loss_function(outs, tars_t.permute(1,0,2))                  

            #     single_loss.backward()

            #     with torch.no_grad():
            #         single_losses += single_loss

    def plot_losses(self, losses, plot_path):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('epochs')
        axes.set_ylabel('loss')
        axes.set_title('History of MSELoss during training')

        
        plt.savefig(f'{plot_path}_losses.png')
        plt.savefig(f'{plot_path}_losses.pdf')
        plt.show()


    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model was saved in: {path}')


def main():
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    ##### Dataset and DataLoader #####
    dataset = TimeSeriesDataset(interaction_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Number of samples: {len(dataset)}")
    
    
    ##### Model parameters #####
    batch_size = 1
    epochs = 5
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    lr = 0.01
    weight_decay = 0.9
    betas = (0.9, 0.999)
    
    hidden_num = 100
    layer_norm = False
    n_dim = 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    
    model_name = f"core_lstm_{n_dim}_{n_features}_{n_independent}_{hidden_num}_{criterion}_{lr}_{weight_decay}_{epochs}"
    model_name += 'lnorm' if layer_norm else ''
    
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    prepro = Preprocessor(num_features=n_features, num_dimensions=n_dim)
    
    trainer = LSTM_Trainer(loss_function=criterion,
                           learning_rate=lr,
                           betas=betas,
                           weight_decay=weight_decay,
                           batch_size=batch_size,
                           hidden_num=hidden_num,
                           layer_norm=layer_norm,
                           num_dim=n_dim,
                           num_feat=n_features,
                           independent_feat=n_independent)

    # Train LSTM
    losses = trainer.train(epochs, dataloader, model_save_path)
    loss_path = f"CoreLSTM/testing_predictions/train_loss/{model_name}.pt"
    trainer.plot_losses(losses, loss_path)
    torch.save(losses, loss_path)
    
    
if __name__ == '__main__':
    main()