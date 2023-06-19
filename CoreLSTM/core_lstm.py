"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch._C import device
from torch.utils.data import DataLoader
from numpy.random import randint

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)   

from Data_Preparation.data_preparation import Preprocessor
from Data_Preparation.interaction_dataset import TimeSeriesDataset

class CORE_NET(nn.Module):
    """
    LSTM core model for Active Tuning:
        -> LSTMCell
        -> if layer_norm=True: Layer Normalization
        -> linear layer
    """

    def __init__(
        self, 
        input_size=27, # 3*6 features + 3 distances + 2 motor response + 4 interaction code
        embedding_size=128,
        hidden_layer_size=360,
        output_size=18, # 3*6 features
        num_interactions=4,
        layer_norm=False
    ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        print(f'DEVICE Core_Net: {self.device}')

        super(CORE_NET,self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_size
        self.layer_norm = layer_norm
        self.num_interactions = num_interactions
        
        self.event_codes = nn.Linear(
            in_features=self.num_interactions,
            out_features=self.num_interactions,
            device=self.device
        )
        
        self.embedding_layer = nn.Linear(
            in_features=self.input_size - self.num_interactions,
            out_features=self.embedding_size,
            device=self.device
        )
        
        self.lstm = nn.LSTMCell(
            input_size=self.embedding_size + self.num_interactions, 
            hidden_size=self.hidden_size, 
            bias=True, 
            device=self.device
            )

        if self.layer_norm:
            self.lnorm = nn.LayerNorm(
                self.hidden_size, 
                device=self.device
            )

        self.linear = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.output_size,  
            device=self.device
        )


    def forward(self, input_seq, interaction_label, state=None):
        
        interaction_label = (interaction_label + randint(0, 4)) % 4
        
        # One hot interaction labels
        one_hot_vector = F.one_hot(
            interaction_label, 
            num_classes=self.num_interactions).to(self.device).to(torch.float32)
        
        # softmax
        softmax = F.softmax(one_hot_vector, dim=1)
        
        # Event code embedding
        interaction_code = F.relu(self.event_codes(softmax))
        
        # Feature embedding (without event code)
        input_seq = F.relu(self.embedding_layer(input_seq))
        
        # Concat embedded code and features to one input
        input_seq = torch.cat([input_seq, interaction_code], dim=1) # shape (batch x n_features+interaction_code)
        
        # LSTM forward pass
        hn, cn = self.lstm(input_seq, state)

        # Linear output layer with optional normalization
        if self.layer_norm:
            prediction = self.linear(self.lnorm(hn))
        else: 
            prediction = self.linear(hn)

        return prediction, (hn,cn)
 
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size).to(self.device),
                torch.zeros(batch_size, self.hidden_size).to(self.device))

    
    def reset_hidden_state(self, state):
        batch_size = state[0].size()[0]
        return (torch.zeros(batch_size, self.hidden_size).to(self.device),
                state[1])
        
    def restructure_data(self, seq, label, interaction):
        seq, label, interaction = seq.to(self.device), label.to(self.device), interaction.to(self.device)
        interaction = interaction.to(torch.int64)
        seq = seq.permute(1,0,2)
        label = label.permute(1,0,2)
        return seq, label, interaction


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE TrainM: {device}')
    
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
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Number of samples: {len(dataset)}")
    
    example = next(iter(dataloader))
    seq, label, interaction = example
    seq, label, interaction = seq.to(device), label.to(device), interaction.to(device)
    interaction = interaction.to(torch.int64)
    
    seq = seq.permute(1,0,2)
    label = label.permute(1,0,2)
    seq_len, batch_size, num_features = seq.size()
    
    
    print(seq.shape, label.shape)
    print(seq.size()[0])
    print(interaction)
    
    mse = nn.MSELoss()
    
    model = CORE_NET(
        input_size=6*3+2+3+4, 
        hidden_layer_size=10, 
        layer_norm=False
        ).to(device)
    
    state = model.init_hidden(batch_size=batch_size)
    print(f"state: {state[0].shape} {state[1].shape}")
    
    outs = []
    print(seq_len)
    
    for j in range(seq_len):
        _input = seq[j, :, :].to(device)
        # print(f"input shape step {j}: {_input.shape}")
        out, state = model.forward(input_seq=_input, interaction_label=interaction, state=state)
        outs.append(out)
        
    outs = torch.stack(outs).to(device)
    print(f"Stacked outputs: {outs.shape}")
    
    loss = mse(outs, label)
    print(f"loss: {loss}")
    
if __name__ == "__main__":
    main()