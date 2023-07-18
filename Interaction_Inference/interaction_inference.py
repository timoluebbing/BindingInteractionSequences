import torch
import sys
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader, random_split
from torch.nn import PairwiseDistance

pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)
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
    criterion : function
        Criterion for comparison of a list of past predictions and a list of
        observations.
    optimizer : torch.optim.Optimizer
        Optimizer to optimize the unfrozen layers during retrospective inference
    inference_steps : int
        Number of repeating steps to perform interaction inference
    num_dim : int
        Number of dimensions for each object
    num_obj : int
        Number of objects / features
    teacher_forcing_steps : int
        Number of timesteps where teacher forcing forward passes are performed
    teacher_forcing_dropouts : bool
        Flag indicating whether dropouts are present during teacher forcing forward passes
    reset_optimizer : bool
        If True the optimizer's statistics are reset before each inference.
        If False the optimizer's statistics are kept across inferences.
    """

    def __init__(
            self, 
            model, 
            criterion,
            optimizer, 
            inference_steps,
            num_dim,
            num_obj,
            teacher_forcing_steps,
            teacher_forcing_dropouts,
            reset_optimizer=False
        ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.inference_steps = inference_steps
        self.reset_optimizer = reset_optimizer

        if self.reset_optimizer:
            self._optimizer_orig_state = optimizer.state_dict()

        self.num_dim = num_dim
        self.num_obj = num_obj

        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_dropouts = teacher_forcing_dropouts
        self.dropout_chance = 0.0
        random.seed(0)
        self.random_thresholds = random.random_sample((self.teacher_forcing_steps,))

        # self.model_state = model_state
        # for s in self.opt_accessor(self.model_state):
        #     s.requires_grad_()
        
        # params_to_train = ['event_codes.weight', 'event_codes.bias']
        for name, param in self.model.named_parameters():
            # Set True only for params in the list 'params_to_train'
            param.requires_grad = False # name in params_to_train


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
        motor_column = self.num_dim*self.num_obj
        motor_force = seq[j, :, motor_column:motor_column+2].squeeze()
        return torch.cat([output, motor_force, distances], dim=1)


    def calculate_new_distances(self, out: torch.Tensor):
        # shape out: [seq_len=1, batch_size, features]
        t = out.squeeze()
        # a1, a2, b = t[:, [0,1]], t[:, [6,7]], t[:, [12,13]]
        if self.num_dim == 4:
            a1, a2, b = (
                t[:, [2*i + i*(self.num_dim-2), 2*(i+1) + i*(self.num_dim-2)]] 
                for i in range(self.num_obj)
            )
            
        elif self.num_dim == 2:
            a1, a2, b = (
                t[:, 2*i : 2*(i+1)]
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
        self.reset_dropout_chance()

        for j in range(seq_len):
            _, state, outs = self.forward_pass(seq, interaction, state, outs, j)
                    
        output = torch.stack(outs).to(self.device)
        
        # hier noch irgendwie output ausm softmax rauslesen
        # und als predicted interaction_code_inference abspreichern und auch returnen
        event_inference = self.model.event_code
        
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
        accuracies = []
        losses = []
        inference_losses = []
        
        seq, label, interaction = self.model.restructure_data(seq, label, interaction)

        for i in range(self.inference_steps):
            
            self.optimizer.zero_grad()
            # self.model.zero_grad() brauch ich nicht, da gradienten des models alle aus
                                    
            # Predict interaction sequence
            output, state, int_inference = self.predict(seq, interaction)
            
            # Compute loss 
            loss = self.criterion(output, label)
            
            # Propagate loss back to interaction code
            loss.backward()

            # Update interaction code: model.event_code
            self.optimizer.step()
            
            with torch.no_grad():
                # Check loss with regards to the correct one hot label
                int_one_hot = nn.functional.one_hot(interaction, num_classes=4)
                inference_loss = self.criterion(int_inference, int_one_hot)
                
                inferece_pred = torch.argmax(int_inference, dim=1)
                correct = sum(inferece_pred == interaction)
                accuracy = correct / len(interaction)
                accuracy_percent = 100. * accuracy
                
                losses.append(loss.clone().item())
                inference_losses.append(inference_loss.clone().item())
                accuracies.append(accuracy.clone().item())

                step_loss = loss.clone().item()
                step_inf_loss = inference_loss.clone().item()     
                if i % 5 == 0:
                    print(f"Inference step {i:3}: Loss: {step_loss:.6f} - Int. Inference Loss: {step_inf_loss:.6f} - Acc: {accuracy_percent:.3f}")
            
        return int_inference, interaction, losses, inference_losses, accuracies
    
    
    def plot_losses(self, losses, plot_path, show=True):
        fig = plt.figure()
        axes = fig.add_axes([0.12, 0.1, 0.8, 0.8]) 
        axes.plot(losses[0], 'r')
        axes.plot(losses[1], 'b')
        axes.legend(['Network loss', 'Inference loss'])
        axes.grid(True)
        axes.set_xlabel('inference steps')
        axes.set_ylabel('loss (log scaled)')
        axes.set_yscale('log')
        axes.set_title('History of loss during interaction inference')
        
        plt.savefig(f'{plot_path}_losses.png', dpi=300)
        if show:
            plt.show()
            
    def plot_accuracy(self, acc, plot_path, show=True):
        fig = plt.figure()
        axes = fig.add_axes([0.12, 0.1, 0.8, 0.8]) 
        axes.plot(acc, 'r')
        axes.grid(True)
        axes.set_xlabel('inference steps')
        axes.set_ylabel('Accuracy (%)')
        axes.set_title('Accuracy during interaction inference')
        
        plt.savefig(f'{plot_path}_accuracy.png', dpi=300)
        if show:
            plt.show()
    
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
    batch_size = 280
    timesteps = 121
    seed = 2023
    no_forces = False
    no_forces_no_orientation = True
    no_forces_out = False
    n_out = 12 if (no_forces or no_forces_out) else 18
    n_out = 6 if no_forces_no_orientation else n_out
    
    dataset = TimeSeriesDataset(
        interaction_paths, 
        timesteps=timesteps,
        no_forces=no_forces,
        no_forces_no_orientation=no_forces_no_orientation,
        n_out=n_out,
        use_distances_and_motor=True)
    print(f"Number of samples:      {len(dataset)} \n")
    
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    seq, label, interaction = next(iter(train_dataloader))
    
    ##### Model parameters #####
    hidden_num = 256
    embedding_num = 16
    layer_norm = False
    
    n_dim = 4 if no_forces else 6
    n_dim = 2 if no_forces_no_orientation else n_dim
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    teacher_forcing_steps = 60
    teacher_forcing_dropouts = True

    inference_steps = 50

    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss()
    criterion = mse_loss
    
    # Load pretrained model
    resnet80 = 'core_res_lstm_4_3_5_128_HuberLoss()_0.001_0.0_360_1500_tfs80_tfd_nf_ts121'
    resnet80_best = 'core_res_lstm_4_3_5_360_HuberLoss()_0.001_0.0_280_2000_tfs80_tfd_nf_ts121'
    
    resnet80_best_tuning = 'core_res_lstm_4_3_5_256_0.001_0.0_HuberLoss()_280_3000_tfs80_tfd'
    resnet60_best_tuning = 'core_res_lstm_4_3_5_256_0.001_0.0_HuberLoss()_280_3000_tfs60_tfd'

    resnet60_no_orientation = 'core_res_lstm_2_3_5_256_HuberLoss()_0.001_0.0_270_1500_tfs60_tfd_nfno_ts121'

    model_name = resnet60_no_orientation
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    # model_save_path = f'CoreLSTM/models/tuning/{model_name}.pt'
    
    model = CORE_NET(
        input_size=n_dim*n_features+n_independent+n_interactions, 
        batch_size=batch_size,
        hidden_layer_size=hidden_num, 
        embedding_size=embedding_num,
        output_size=n_out,
        layer_norm=layer_norm,
        interaction_inference=True
    )

    model.load_state_dict(torch.load(model_save_path))
    
    params = {model.event_code}

    optimizer = AdamW(
        params=params,
        lr=0.02,
    )

    inference = InteractionInference(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        inference_steps=inference_steps,
        num_dim=n_dim,
        num_obj=n_features,
        teacher_forcing_steps=teacher_forcing_steps,
        teacher_forcing_dropouts=teacher_forcing_dropouts,
    )
    
    print(f"model.event_code.requires_grad = {model.event_code.requires_grad}")

    int_inference, interaction, losses, inference_losses, accuracies = inference.infer_interactions(
        # dataloader=train_dataloader
        seq=seq,
        label=label,
        interaction=interaction,
    )
    
    with torch.no_grad():
        print(int_inference[:5])
        print(torch.round(F.softmax(int_inference[:5] * 10, dim=1), decimals=2))
        print(interaction[:5])
        
        inferece_pred = torch.argmax(int_inference, dim=1)
        correct = sum(inferece_pred == interaction)
        accuracy = correct / len(interaction)
        accuracy_percent = 100. * accuracy
        
        print("\nFinal inference results: \n")
        print(f"Correct / Total = {correct} / {len(interaction)} = {accuracy:.4f}")
        print(f"Inference accuracy: {accuracy_percent:.4f} %")
        print(f"Inference accuracy: {accuracies[-1]:.4f}")
        
        plot_path   = f"CoreLSTM/testing_predictions/inference/{model_name}"
        
        inference.plot_accuracy(accuracies, plot_path)
        inference.plot_losses([losses, inference_losses], plot_path)
        
        
    
        
        
        
        
        
    
if __name__ == "__main__":
    main()