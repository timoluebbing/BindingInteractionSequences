import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.train_core_lstm import LSTM_Trainer
from CoreLSTM.test_core_lstm import LSTM_Tester
from Data_Preparation.interaction_dataset import TimeSeriesDataset


def main(train=True, validate=True, test=True, render=False):
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    no_forces = True
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    ##### Dataset and DataLoader #####
    seed = 2023
    no_forces = True
    batch_size = 180

    dataset = TimeSeriesDataset(
        interaction_paths, 
        no_forces=no_forces,
        n_out=12,
        use_distances_and_motor=True
    )
    generator = torch.Generator().manual_seed(seed)
    #split = [0.7, 0.15, 0.15]
    split = [0.6, 0.3, 0.1]
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, split, generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    
    ##### Model parameters #####
    epochs = 2000
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    lr = 0.0001
    weight_decay = 0.01
    betas = (0.9, 0.999)
    teacher_forcing_steps = 200
    teacher_forcing_dropouts = False
    
    hidden_num = 360
    layer_norm = True

    n_dim = 4 if no_forces else 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    model_name = f"core_lstm_{n_dim}_{n_features}_{n_independent}_{hidden_num}_{criterion}_{lr}_{weight_decay}_{batch_size}_{epochs}"
    model_name += '_lnorm' if layer_norm else ''
    model_name += f'_tfs{teacher_forcing_steps}'
    model_name += '_tfd' if teacher_forcing_dropouts else ''
    
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
        

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
            num_interactions=n_interactions
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
        
        test_loss = trainer.validate(test_dataloader)
        print(test_loss)
        

    if test:
        model_path = 'CoreLSTM/models/core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_400_lnorm_tfs200.pt'
        current_best = 'CoreLSTM/models/core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200.pt'
        
        tester = LSTM_Tester(
            loss_function=criterion,
            batch_size=batch_size,
            hidden_num=hidden_num,
            layer_norm=layer_norm,
            num_dim=n_dim,
            num_feat=n_features,
            num_independent_feat=n_independent,
            num_interactions=n_interactions,
            model_save_path=model_save_path, 
            # model_save_path=model_path, 
            # model_save_path=current_best,  
        )
        
        print("Test dataset: \n")
        _ = tester.evaluate(test_dataloader)

        total_loss, losses, obj_losses, type_losses = tester.evaluate_detailed(test_dataloader)
        print(total_loss)
        print(sum(losses))
        print(np.sum(obj_losses, axis=None))
        print(np.sum(type_losses, axis=None))
        
        if render:
        # Check prediction for one example with renderer
            tester.evaluate_model_with_renderer(
                test_dataloader, 
                # train_dataloader,
                n_samples=5
            )
    
if __name__ == '__main__':
    main()