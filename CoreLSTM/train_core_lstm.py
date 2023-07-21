
import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from numpy import random
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from torch.optim import Adam, AdamW, SGD
from torch.nn import PairwiseDistance

from tqdm import tqdm
import sys
import copy
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET


class LSTM_Trainer():

    """
        Class to train core LSTM model on interactions sequences.
        
    """

    def __init__(
        self, 
        loss_function, 
        learning_rate, 
        betas, 
        weight_decay, 
        batch_size, 
        hidden_num,
        teacher_forcing_steps,
        teacher_forcing_dropouts, 
        layer_norm, 
        num_dim, 
        num_feat,
        num_independent_feat,
        num_interactions,
        num_output,
        pretrained_path=None,
    ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 
        print(f'DEVICE TrainM: {self.device}')
        
        self.num_dim = num_dim
        self.num_feat = num_feat
        self.num_independent_feat = num_independent_feat
        self.num_interactions = num_interactions
        self.num_output = num_output
        self.hidden_num = hidden_num
        self.layer_norm = layer_norm
        self.pretrained_path = pretrained_path

        self.model = CORE_NET(
            input_size=num_dim*num_feat+num_independent_feat+num_interactions, 
            batch_size=batch_size,
            hidden_layer_size=hidden_num, 
            output_size=num_output,
            layer_norm=layer_norm
        )
        
        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(self.pretrained_path))
            
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_dropouts = teacher_forcing_dropouts
        self.dropout_chance = 0.0
        # random.seed(0)
        self.random_thresholds = random.random_sample((teacher_forcing_steps,))
        
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay
        )

        print('Initialized model!')
        print(self.model)
        print(self.loss_function)
        print(self.optimizer)


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay


    def set_loss_function(self, loss_function):
        self.loss_function = loss_function


    def set_teacher_forcing_steps(self, teacher_forcing_steps):
        self.teacher_forcing_steps = teacher_forcing_steps


    def set_hidden_num(self, hidden_num):
        self.hidden_num = hidden_num


    def set_layer_norm(self, layer_norm):
        self.layer_norm = layer_norm


    def reset_model(self):
        self.model = CORE_NET(
            input_size=self.num_dim*self.num_feat+self.num_independent_feat+self.num_interactions, 
            batch_size=self.batch_size,
            hidden_layer_size=self.hidden_num, 
            output_size=self.num_output,
            layer_norm=self.layer_norm
        )


    def reset_optimizer(self):
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


    def set_tuning_params(self, params: list):
        setters = [
            self.set_hidden_num,
            self.set_layer_norm,
            self.set_learning_rate,
            self.set_weight_decay,
            self.set_loss_function,
            self.set_teacher_forcing_steps
        ]
        assert len(params) == len(setters)

        for setter, param in zip(setters, params):
            setter(param)

        self.reset_model()
        self.reset_optimizer()


    def set_tuning_model_name(self, epochs):
        model_name = f"core_res_lstm_{self.num_dim}_{self.num_feat}_{self.num_independent_feat}_{self.hidden_num}"
        model_name += '_lnorm' if self.layer_norm else ''
        model_name += f"_{self.learning_rate}_{self.weight_decay}_{self.loss_function}_{self.batch_size}_{epochs}"
        model_name += f'_tfs{self.teacher_forcing_steps}'
        model_name += '_tfd' if self.teacher_forcing_dropouts else ''
        return f"{model_name}.pt"


    def tuning_module(
        self, 
        params,
        epochs,
        save_path,
        train_dataloader,
        validate,
        validate_dataloader,
        display_best_n_combinations=2,
    ):
        hyperparams = params.values()
        combinations = list(it.product(*hyperparams))

        all_train_losses, all_val_losses = [], []
        min_val_losses = np.zeros(len(combinations))

        for i, combination in enumerate(combinations):
            
            self.set_tuning_params(combination)
            
            model_save_path = save_path + self.set_tuning_model_name(epochs)
            
            print(f"Hyperparameter tuning iteration {i}: Params: {combination}")
            print(f"Model save path: {model_save_path}")

            train_losses, val_losses = self.train_and_validate(
                epochs=epochs,
                save_path=model_save_path,
                train_dataloader=train_dataloader,
                validate=validate,
                val_dataloader=validate_dataloader
            )

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

            min_val_losses[i] = min(val_losses)
        
        min_idx = np.argsort(min_val_losses)[:display_best_n_combinations]
        best_min_val_losses = min_val_losses[min_idx]#[:display_best_n_combinations]]
        best_combinations = np.array(combinations)[min_idx]#[:display_best_n_combinations]]

        print("Hpyerparameter tuning finished!")
        print("Results summary: \n\n")
        print(f"Validation loss: | Params: {list(params.keys())}")
        for (loss, param) in zip(best_min_val_losses, best_combinations):
            print(f"{loss:.13f}  | {param.tolist()}")

        return all_train_losses, all_val_losses, best_min_val_losses, best_combinations, min_idx


    def train_and_validate(self, 
              epochs, 
              save_path, 
              train_dataloader,
              validate=False,
              val_dataloader=None,         
        ):

        train_losses = []
        val_losses = []
        min_loss = float('inf')
        n_last_epochs = epochs / 10

        for epoch in range(epochs):

            ep_trn_loss = self.train_epoch(epoch, train_dataloader)
            train_losses.append(ep_trn_loss)
            
            if validate and epoch >= epochs-n_last_epochs:
                ep_val_loss = self.validate(val_dataloader, epoch)
                val_losses.append(ep_val_loss)
                
                if ep_val_loss < min_loss:
                    min_loss = ep_val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
        model_state = best_model_wts if validate else self.model.state_dict()
        self.save_model(save_path, model_state)
        print(f"Model with minimum validation loss: {min(val_losses):.6f}\n")

        return train_losses, val_losses
    
    def train_epoch(self, epoch, dataloader):
        
        losses = torch.tensor([0.0], device=self.device)

        self.model.zero_grad()
        self.optimizer.zero_grad()

        for seq, label, interaction in dataloader:
            
            seq, label, interaction = self.model.restructure_data(seq, label, interaction)
            
            outs = self.predict_sequence(
                seq, 
                interaction, 
            )

            loss = self.loss_function(outs, label)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                losses += loss
        
        with torch.no_grad():

            ep_loss = losses.clone().item()
            if epoch % 20  == 0:
                avg_loss = ep_loss /  (len(dataloader) * self.batch_size)
                print(f'Epoch: {epoch:4} - Avg. Loss: {avg_loss:10.8f} - Epoch Loss: {ep_loss:8.4f} - LR: {self.get_lr(self.optimizer):.6f}')

        return ep_loss
    

    def predict_sequence(
        self, 
        seq, 
        interaction
    ):
        """Method to pridict a single interaction sequence. Wether training is open or 
           closed loop is determined by the number of teacher forcing steps and a 
           teacher forcing dropout probability (currently increasing linearly with every 
           timestep as long as teacher forcing is enabled)

        Args:
            seq (torch.Tensor): Input sequences
            interaction (torch.Tensor): Interaction labels

        Returns:
            (torch.Tensor, torch.Tensor): Returns the output for the given sequence and 
                                          the cumulated new loss
        """        
        seq_len, batch_size, _ = seq.size()

        state = self.model.init_hidden(batch_size=batch_size)
        outs = []
        self.reset_dropout_chance()

        for j in range(seq_len):
            _, state, outs = self.forward_pass(seq, interaction, state, outs, j)

        outs = torch.stack(outs)

        return outs


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
        motor_column = self.num_dim*self.num_feat
        motor_force = seq[j, :, motor_column:motor_column+2].squeeze()
        return torch.cat([output, motor_force, distances], dim=1)


    def calculate_new_distances(self, out: torch.Tensor):
        # shape out: [seq_len=1, batch_size, features]
        t = out.squeeze()
        # a1, a2, b = t[:, [0,1]], t[:, [6,7]], t[:, [12,13]]
        if self.num_dim == 2:
            a1, a2, b = (
                t[:, 2*i : 2*(i+1)]
                for i in range(self.num_feat)
            )

        elif self.num_dim in [4, 6]:
            a1, a2, b = (
                t[:, [2*i + i*(self.num_dim-2), 2*(i+1) + i*(self.num_dim-2)]] 
                for i in range(self.num_feat)
            )

        # abstrahieren für n_obj!=3 mit [a1, a2, b, ...] als list und:
        # combs = it.combinations(tensors, 2)
        # dann loop über combs und dist anwenden

        dist = PairwiseDistance(p=2)
        dis_a1_a2 = dist(a1, a2)
        dis_a1_b  = dist(a1, b)
        dis_b_a2  = dist(b, a2)

        return torch.stack([dis_a1_a2, dis_a1_b, dis_b_a2], dim = 1) # shape (batchsize x 3)
    

    def validate(self, dataloader, epoch):
        
        self.model.eval()
        
        with torch.no_grad():

            loss = torch.tensor([0.0], device=self.device)
            
            for seq, label, interaction in dataloader:
                    
                seq, label, interaction = self.model.restructure_data(seq, label, interaction)

                seq_len, batch_size, num_features = seq.size()
        
                state = self.model.init_hidden(batch_size=batch_size)
                outs = []
                self.reset_dropout_chance()
                
                for j in range(seq_len):
                    _, state, outs = self.forward_pass(seq, interaction, state, outs, j)
                    
                outs = torch.stack(outs)
                
                single_loss = self.loss_function(outs, label)
                loss += single_loss
                
            ep_loss = loss.clone().item()
            if epoch % 5 == 0:
                avg_loss = ep_loss / (len(dataloader) * self.batch_size)
                print(f'Validate:     Avg. loss: {avg_loss:10.8f} - Total loss: {ep_loss:8.4f}')
                
            return ep_loss
    

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    

    def plot_losses(self, losses, plot_path, show=True):
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
        if show:
            plt.show()


    def save_model(self, path, model_state):
        torch.save(model_state, path)
        print(f'Model was saved in: {path}')