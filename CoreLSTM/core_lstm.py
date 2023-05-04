"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import torch
from torch import nn
from torch._C import device

class CORE_NET(nn.Module):
    """
    LSTM core model for Active Tuning:
        -> LSTMCell
        -> if layer_norm=True: Layer Normalization
        -> linear layer
    """

    def __init__(
        self, 
        input_size=15, 
        hidden_layer_size=360, 
        layer_norm=False
    ):

        self.device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 
        print(f'DEVICE TestM: {self.device}')

        super(CORE_NET,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.layer_norm = layer_norm
        
        self.lstm = nn.LSTMCell(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            bias=True, 
            device=self.device)

        if self.layer_norm:
            self.lnorm = nn.LayerNorm(
                self.hidden_size, 
                device=self.device
            )

        self.linear = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.input_size, 
            device=self.device
        )


    def forward(self, input_seq, state=None):
                
        hn, cn = self.lstm(input_seq, state)

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
