import torch 
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.train_core_lstm import LSTM_Trainer
from Data_Preparation.interaction_dataset import TimeSeriesDataset


def main(train=False, validate=True, test=True):
    
    seed = 0
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    ##### Dataset and DataLoader #####
    batch_size = 180

    dataset = TimeSeriesDataset(interaction_paths, use_distances_and_motor=True)
    generator = torch.Generator().manual_seed(seed)
    split = [0.7, 0.15, 0.15]
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, split, generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    
    ##### Model parameters #####
    epochs = 2000
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    lr = 0.0001
    weight_decay = 0
    betas = (0.9, 0.999)
    teacher_forcing_steps = 200
    teacher_forcing_dropouts = False
    
    hidden_num = 360
    layer_norm = True

    n_dim = 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    
    model_name = f"core_lstm_{n_dim}_{n_features}_{n_independent}_{hidden_num}_{criterion}_{lr}_{weight_decay}_{batch_size}_{epochs}"
    model_name += '_lnorm' if layer_norm else ''
    model_name += f'_tfs{teacher_forcing_steps}'
    model_name += '_tfd' if teacher_forcing_dropouts else ''
    
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    # prepro = Preprocessor(num_features=n_features, num_dimensions=n_dim)
    
    trainer = LSTM_Trainer(loss_function=criterion,
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
                           independent_feat=n_independent)

    # Train LSTM
    if train:
        train_losses, val_losses = trainer.train_and_validate(epochs, 
                                                              model_save_path, 
                                                              train_dataloader, 
                                                              validate,
                                                              val_dataloader)
        
        train_loss_path = f"CoreLSTM/testing_predictions/train_loss/{model_name}"
        val_loss_path   = f"CoreLSTM/testing_predictions/val_loss/{model_name}"
        trainer.plot_losses(train_losses, train_loss_path)
        trainer.plot_losses(val_losses,   val_loss_path)
        # torch.save(losses, loss_path)
        
        if test:
            print("Test dataset: \n")
            _ = trainer.evaluate(test_dataloader)
    
    # Check prediction for one example with renderer
    model_path = 'CoreLSTM/models/core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_400_lnorm_tfs200.pt'
    current_best = 'CoreLSTM/models/core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2000_lnorm_tfs200.pt'
    trainer.evaluate_model_with_renderer(# test_dataloader, 
                                         train_dataloader,
                                         # model_path,
                                         current_best,
                                         # model_save_path, 
                                         n_samples=5)
    
if __name__ == '__main__':
    main()