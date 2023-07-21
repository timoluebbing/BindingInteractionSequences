import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)      
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.train_core_lstm import LSTM_Trainer
from CoreLSTM.test_core_lstm import LSTM_Tester
from Data_Preparation.interaction_dataset import TimeSeriesDataset


def main(train=True, validate=True, test=True, render=True):
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    
    ##### Dataset and DataLoader #####
    seed = 2023
    batch_size = 270 # * 3 = 840 = train_size
    timesteps = 121
    no_forces = False
    no_forces_no_orientation = False
    no_forces_out = False
    n_out = 12 if (no_forces or no_forces_out) else 18
    n_out = 6 if no_forces_no_orientation else n_out

    dataset = TimeSeriesDataset(
        interaction_paths, 
        timesteps=timesteps,
        no_forces=no_forces,
        no_forces_no_orientation=no_forces_no_orientation,
        n_out=n_out,
        use_distances_and_motor=True
    )
    generator = torch.Generator().manual_seed(seed)
    split = [0.7, 0.15, 0.15]
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, split, generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    
    ##### Model parameters #####
    epochs = 1500
    
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss()
    criterion = huber_loss
    lr = 0.001
    weight_decay = 0.0 # 0.01
    betas = (0.9, 0.999)
    teacher_forcing_steps = 60
    teacher_forcing_dropouts = True
    
    hidden_num = 256
    layer_norm = False

    n_dim = 4 if no_forces else 6
    n_dim = 2 if no_forces_no_orientation else n_dim
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    model_name = f"core_res_lstm_{n_dim}_{n_features}_{n_independent}_{hidden_num}_{criterion}_{lr}_{weight_decay}_{batch_size}_{epochs}"
    model_name += '_lnorm' if layer_norm else ''
    model_name += f'_tfs{teacher_forcing_steps}'
    model_name += '_tfd' if teacher_forcing_dropouts else ''
    model_name += '_nf' if no_forces else ''
    model_name += '_nfno' if no_forces_no_orientation else ''
    model_name += '_nfo' if no_forces_out else ''
    model_name += f'_ts{timesteps}'
    
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    # load pretrained model for further training    
    resnet120 = 'core_res_lstm_4_3_5_128_HuberLoss()_0.001_0.0_360_1500_tfs120_tfd_nf_ts121'
    pretrained_path = f'CoreLSTM/models/{resnet120}.pt'
    
    if train:
        
        trainer = LSTM_Trainer(
            loss_function=criterion,
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay,
            batch_size=batch_size,
            hidden_num=hidden_num,
            teacher_forcing_steps=teacher_forcing_steps,
            teacher_forcing_dropouts=teacher_forcing_dropouts,
            layer_norm=layer_norm,
            num_dim=n_dim,
            num_feat=n_features,
            num_independent_feat=n_independent,
            num_interactions=n_interactions,
            num_output=n_out,
            # pretrained_path=pretrained_path,
        )
        
        train_losses, val_losses = trainer.train_and_validate(
            epochs, 
            model_save_path, 
            train_dataloader, 
            validate,
            val_dataloader
        )
        
        train_loss_path = f"CoreLSTM/testing_predictions/train_loss/{model_name}"
        val_loss_path   = f"CoreLSTM/testing_predictions/val_loss/{model_name}"
        trainer.plot_losses(train_losses, train_loss_path)
        trainer.plot_losses(val_losses,   val_loss_path)        


    if test:
        
        tester = LSTM_Tester(
            loss_function=criterion,
            batch_size=batch_size,
            hidden_num=hidden_num,
            layer_norm=layer_norm,
            timesteps=timesteps,
            teacher_forcing_steps=teacher_forcing_steps,
            teacher_forcing_dropouts=teacher_forcing_dropouts,
            num_dim=n_dim,
            num_feat=n_features,
            num_independent_feat=n_independent,
            num_interactions=n_interactions,
            num_output=n_out,
            model_save_path=model_save_path,   
        )
        
        print("Test dataset: \n")
        _ = tester.evaluate(test_dataloader)

        total_loss, losses, obj_losses, type_losses = tester.evaluate_detailed(test_dataloader)
        print(total_loss)
        print(sum(losses))
        print(np.sum(obj_losses, axis=None))
        print(np.sum(type_losses, axis=None))
        
        if render:
        # Check prediction for n examples with renderer
            tester.evaluate_model_with_renderer(
                test_dataloader, 
                # train_dataloader,
                n_samples=5
            )
    
if __name__ == '__main__':
    main()