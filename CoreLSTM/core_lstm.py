import torch
from torch import nn, tanh
import torch.nn.functional as F


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
        embedding_size=32,
        embedding_size2=64,
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
        self.embedding_size2 = embedding_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_size
        self.layer_norm = layer_norm
        self.num_interactions = num_interactions
        
        self.interaction_embedding = nn.Linear(
            in_features=self.num_interactions,
            out_features=self.num_interactions,
            device=self.device
        )
        
        self.input_embedding = nn.Linear(
            in_features=self.input_size - self.num_interactions,
            out_features=self.embedding_size,
            device=self.device
        )
        
        self.combined_embedding = nn.Linear(
            in_features=self.embedding_size + self.num_interactions,
            out_features=self.embedding_size2,
            device=self.device
        )        
        
        self.lstm = nn.LSTMCell(
            input_size=self.embedding_size2, 
            hidden_size=self.hidden_size, 
            bias=True, 
            device=self.device
            )

        if self.layer_norm:
            self.lnorm = nn.LayerNorm(
                self.hidden_size, 
                device=self.device
            )

        self.fc1 = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size // 2,  
            device=self.device
        )
        
        self.fc2 = nn.Linear(
            in_features=self.hidden_size // 2, 
            out_features=self.output_size,  
            device=self.device
        )

    def forward(self, input_seq, interaction_label, state=None):
        
        # interaction_label = (interaction_label + randint(0, 4)) % 4
        
        # One hot interaction labels
        one_hot_vector = F.one_hot(
            interaction_label, 
            num_classes=self.num_interactions).to(self.device).to(torch.float32)
        
        # softmax
        softmax = F.softmax(one_hot_vector, dim=1)
        
        # Event code embedding
        interaction_code = F.relu(self.interaction_embedding(softmax))
        
        # Feature embedding (without event code)
        _input = F.relu(self.input_embedding(input_seq))
        
        # Concat embedded code and features to one input
        _input = torch.cat([_input, interaction_code], dim=1) # shape (batch x n_features+interaction_code)
        
        # Final combined embedding for concatenated features and event code
        _input = F.relu(self.combined_embedding(_input))
        
        # LSTM forward pass
        hn, cn = self.lstm(_input, state)

        # MLP with non-linear tanh activation for prediction with optional normalization
        if self.layer_norm:
            hn_non_linear = tanh(self.fc1(self.norm(hn)))
            prediction    = tanh(self.fc2(self.lnorm(hn_non_linear)))
        else:
            hn_non_linear = tanh(self.fc1(hn))
            prediction    = tanh(self.fc2(hn_non_linear))

        # Residual network: prediction + _input[:self.output_size]
        input_for_resnet = input_seq[:,:self.output_size]
        
        return prediction + input_for_resnet, (hn,cn)
 
    
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