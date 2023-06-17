import torch
import sys
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader, random_split#

pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET
from Data_Preparation.interaction_dataset import TimeSeriesDataset


class InteractionInference():
    """
    Interaction inference class.

    An instance of this class provides the functionality to infer interactions.

    Parameters
    ----------
    model : torch.nn.Module
        Recurrent neural network model which might be pretrained.
    initial_model_states : torch.Tensor or tuple
        Initial hidden (and cell) state of the model.
    opt_accessor : function
        Function that returns list of tensors to be optimized
    criterion : function
        Criterion for comparison of a list of past predictions and a list of
        observations.
    optimizer : torch.optim.Optimizer
        Optimizer to optimize the unfrozen layers during retrospective inference
    reset_optimizer : bool
        If True the optimizer's statistics are reset before each inference.
        If False the optimizer's statistics are kept across inferences.
    
    # context_handler : function
    #     Function that is applied to the context after each optimization,
    #     e.g. can be used to keep context in certain range.

    """

    def __init__(
            self, 
            model, 
            # model_state, 
            opt_accessor, 
            # context, 
            criterion,
            optimizer, 
            inference_steps,
            reset_optimizer=True, 
            # context_handler=lambda x: x
        ):
        
        self.model = model
        # self.model_state = model_state
        self.opt_accessor = opt_accessor
        # self.context = context
        # self.context.requires_grad = True
        self.criterion = criterion
        self.optimizer = optimizer
        self.inference_steps = inference_steps
        self.reset_optimizer = reset_optimizer
        # self._context_handler = context_handler

        if self.reset_optimizer:
            self._optimizer_orig_state = optimizer.state_dict()

        # # Buffers to store the histories of inputs and outputs
        # self._model_inputs = []
        # self._observations = []

        # self.model_state = model_state
        # for s in self.opt_accessor(self.model_state):
        #     s.requires_grad_()
        
        params_to_train = ['event_codes.weight', 'event_codes.bias']
        for name, param in self.model.named_parameters():
            # Set True only for params in the list 'params_to_train'
            param.requires_grad = name in params_to_train

        # Number of tensors with gradients selected for 
        # the optimizer equals number of selected tensors with _opt_accessor
        # assert (len(self._opt_accessor(self.model_state)) ==
        #         len(self.optimizer.param_groups[1]['params']))

    def predict(self, seq, interaction):
        """
        Predict from the past.

        Predict observations given past inputs as well as an initial hidden
        state and a context.

        Parameters
        ----------
        seq   : torch.Tensor
            interaction sequence
        interaction : torch.Tensor [1 x batchsize]
            interaction label indicating event code
        state : torch.Tensor or tuple
            Initial hidden (and cell) state of the network.

        Returns
        -------
        outputs : tensor
            Entire predicted interaction sequence
        states : tensor
            Last hidden state of the network
        """
        
        seq_len, batch_size, _ = seq.size()

        state = self.model.init_hidden(batch_size=batch_size)
        outs = []

        for j in range(seq_len):
            _input = seq[j, :, :].to(self.device)
            out, state = self.model.forward(input_seq=_input, interaction_label=interaction, state=state)
            outs.append(out)
            
        output = torch.stack(outs).to(self.device)
        
        # hier noch irgendwie output ausm softmax rauslesen
        # und als predicted interaction_code_inference abspreichern und auch returnen
        event_inference = None
        
        return output, state, event_inference
    

    def infer_interactions(self, seq, label, interaction):
        """
        Optimize the current context.

        Given current inputs and observations of the entire event sequence, 
        this method infers the interaction event codes.

        Parameters
        ----------
        seq : torch.Tensor
            input sequence.
        label : torch.Tensor
            target sequence.
        interaction : torch.Tensor
            target event codes.

        Returns
        -------
        context : torch.Tensor
            Optimized context.
        outputs : list
            Past predictions corresponding to the optimized context and
            possibly hidden and/or cell state.
        states : list of torch.Tensor or list of tuple
            Past model states corresponding to the optimized context and
            possibly hidden and/or cell state.

        """
        if self._reset_optimizer:
            self._optimizer.load_state_dict(self._optimizer_orig_state)

        seq, label, interaction = self.model.restructure_data(seq, label, interaction)
        
        for i in range(self.inference_steps):
            
            self.optimizer.zero_grad()
            
            # Predict interaction sequence
            output, state = self.predict(seq, interaction)
            output, state, int_inference = self.predict(seq, interaction)
            
            # Compute loss 
            loss = self.criterion(output, label)
            
            loss.backwards()
            self.optimizer.step()
            
            int_one_hot = nn.functional.one_hot(interaction, num_classes=4)
            inference_loss = self.criterion(int_inference, int_one_hot)
            
            print(f"Inference step {i}: Loss: {loss} - Int. Inference Loss: {inference_loss}")
        
        return output, state, int_inference

    
def main():
    
    seed = 0
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    
    ##### Dataset and DataLoader #####
    batch_size = 180
    timesteps = 201
    seed = 2023
    no_forces = False
    no_forces_out = False
    n_out = 12 if (no_forces or no_forces_out) else 18
    
    dataset = TimeSeriesDataset(
        interaction_paths, 
        timesteps=timesteps,
        no_forces=no_forces,
        n_out=n_out,
        use_distances_and_motor=True)
    print(f"Number of samples:      {len(dataset)} \n")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # generator = torch.Generator().manual_seed(seed)
    
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator)
    
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    
    # print(f"Number of train samples:     {len(train_dataset)}")
    # print(f"Number of valiation samples: {len(val_dataset)}")
    # print(f"Number of test samples:      {len(test_dataset)} \n")
    
    example = next(iter(dataloader))
    seq, label, interaction = example
    seq = seq.permute(1,0,2)
    label = label.permute(1,0,2)
    
    print("\nExample sequence:")
    print(f"Sequence shape:  {seq.shape} {seq.dtype}")
    print(f"Label shape:     {label.shape} {label.dtype}")
    print(f"Interaction:     {interaction} {interaction.dtype}")
    
    ##### Model parameters #####
    hidden_num = 360
    layer_norm = True
    
    n_dim = 4 if no_forces else 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    current_best = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200'
    current_best_dropout = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200_tfd'
    current_best_dropout_wd = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0.01_180_2000_lnorm_tfs200_tfd'
    no_forces_best = 'core_lstm_4_3_5_360_MSELoss()_0.0001_0_180_2500_lnorm_tfs200_nf'
    no_forces_out_best = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2500_lnorm_tfs200_nfo'
    
    model_name = current_best_dropout
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    
    model = CORE_NET(
        input_size=n_dim*n_features+n_independent+n_interactions, 
        hidden_layer_size=hidden_num, 
        output_size=n_out,
        layer_norm=layer_norm
    )
    model.load_state_dict(torch.load(model_save_path))
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=0.01,
    )
    # print(len(optimizer.param_groups[1]['params']))
    
    inference = InteractionInference(
        model=model,
        opt_accessor=None,
        criterion=criterion,
        optimizer=optimizer,
        inference_steps=100,
        reset_optimizer=False,
    )
    
    for name, param in inference.model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
    


if __name__ == "__main__":
    main()
    
    
    
            
            
 
    # def predict(self, state):
    #     outputs = []
    #     states = []
    #     # Forward pass over inference_length steps
    #     for ci_t in range(len(self._model_inputs)):
    #         in_t = self._model_inputs[ci_t]
    #         in_t_c = torch.cat((self._context, in_t), dim=2)
    #         output, state = self._model.forward(in_t_c, state)
    #         outputs.append(output)
    #         states.append(state)
    #     return outputs, states
        
    # def infer_contexts(self, model_input, observation):

    #     # assert (len(model_input.shape) ==
    #     #         3), "model_input should be of shape (seq_len, batch, input_size)"
    #     # assert (model_input.size(0) == 1), "seq_len of model_input should be 1"
    #     # assert (len(observation.shape) ==
    #     #         3), "observation should be of shape (seq_len, batch, input_size)"
    #     # assert (observation.size(0) == 1), "seq_len of observation should be 1"

    #     if self._reset_optimizer:
    #         self._optimizer.load_state_dict(self._optimizer_orig_state)

    #     # # Shift inputs and observations by one
    #     # self._model_inputs.append(model_input)
    #     # self._model_inputs = self._model_inputs[-self._inference_length:]
    #     # self._observations.append(observation)
    #     # self._observations = self._observations[-self._inference_length:]

    #     # Perform context inference cycles
    #     for _ in range(self._inference_cycles):
    #         self._optimizer.zero_grad()

    #         outputs, states = self.predict(self._model_state)

    #         # Compute loss
    #         loss = self._criterion(outputs, self._observations)

    #         # Backward pass
    #         loss.backward()
    #         self._optimizer.step()

    #         # Operations on the data are not tracked
    #         self._context.data = self._context_handler(self._context.data)

    #     # Context and state have been optimized; this optimized context/state
    #     # is now propagated once more in forward direction in order to generate
    #     # the final output and state to be returned
    #     with torch.no_grad():
    #         outputs, states = self.predict(self._model_state)
    #         for i in range(len(self._model_state)):
    #             for j in range(len(self._model_state[i])):
    #                 self._model_state[i][j].data = states[0][i][j].data

    #     return self._context, outputs, states