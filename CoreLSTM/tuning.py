import torch 
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, L1Loss, HuberLoss

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.train_core_lstm import LSTM_Trainer
from Data_Preparation.interaction_dataset import TimeSeriesDataset


def main(validate=True):
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    
    ##### Dataset and DataLoader #####
    seed = 2023
    batch_size = 360
    timesteps=121
    no_forces = True
    no_forces_out = False
    n_out = 12 if (no_forces or no_forces_out) else 18

    dataset = TimeSeriesDataset(
        interaction_paths, 
        timesteps=timesteps,
        no_forces=no_forces,
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
    epochs = 3000

    hidden_num = 360
    layer_norm = True
    n_dim = 4 if no_forces else 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    n_interactions = len(interactions)
    
    mse_loss = MSELoss()
    l1_loss = L1Loss()
    huber_loss = HuberLoss()
    criterion = mse_loss
    lr = 0.0001
    weight_decay = 0 # 0.01
    betas = (0.9, 0.999)
    teacher_forcing_steps = timesteps
    teacher_forcing_dropouts = True

    params = {
        'hidden': [360, 520], # 360
        'lnorm': [False], # False
        'lr': [0.001, 0.0001],
        'wd': [0.01, 0.05],
        'loss': [huber_loss], #l1_loss
        'tf_steps': [80],
    }
    
    model_save_path = 'CoreLSTM/models/tuning/'

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
        num_output=n_out
    )
    
    (train_losses, val_losses, 
     min_val_losses, min_params, best_idx) = trainer.tuning_module(
        params,
        epochs, 
        model_save_path, 
        train_dataloader, 
        validate,
        val_dataloader,
        display_best_n_combinations=5
    )
    
    best_train_losses = train_losses[best_idx[0]]
    best_val_losses = val_losses[best_idx[0]]

    train_loss_path = "CoreLSTM/testing_predictions/tuning/best2_train"
    val_loss_path   = "CoreLSTM/testing_predictions/tuning/best2_val"
    trainer.plot_losses(best_train_losses, train_loss_path, show=False)
    trainer.plot_losses(best_val_losses,   val_loss_path, show=False)        

    
if __name__ == '__main__':
    main()