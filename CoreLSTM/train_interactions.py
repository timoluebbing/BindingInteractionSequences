import torch 
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)      
# Before run: replace ... with current directory path

from CoreLSTM.core_lstm import CORE_NET
from CoreLSTM.train_core_lstm import LSTM_Trainer
from Data_Preparation.data_preparation import Preprocessor
from Data_Preparation.interaction_dataset import TimeSeriesDataset
from Data_Preparation.interaction_renderer import Interaction_Renderer


def main(train=False):
    
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
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    
    ##### Model parameters #####
    epochs = 300
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    lr = 0.0001
    weight_decay = 0
    betas = (0.9, 0.999)
    teacher_forcing_ratio = 1
    
    hidden_num = 360
    layer_norm = True

    n_dim = 6
    n_features = 3
    n_independent = 5 # 2 motor + 3 distances 
    
    model_name = f"core_lstm_{n_dim}_{n_features}_{n_independent}_{hidden_num}_{criterion}_{lr}_{weight_decay}_{batch_size}_{epochs}"
    model_name += '_lnorm' if layer_norm else ''
    model_name += f'_tfr{teacher_forcing_ratio}'
    
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    prepro = Preprocessor(num_features=n_features, num_dimensions=n_dim)
    
    trainer = LSTM_Trainer(loss_function=criterion,
                           learning_rate=lr,
                           betas=betas,
                           weight_decay=weight_decay,
                           batch_size=batch_size,
                           hidden_num=hidden_num,
                           teacher_forcing_ratio=teacher_forcing_ratio,
                           layer_norm=layer_norm,
                           num_dim=n_dim,
                           num_feat=n_features,
                           independent_feat=n_independent)

    # Train LSTM
    if train:
        losses = trainer.train(epochs, train_dataloader, model_save_path)
        loss_path = f"CoreLSTM/testing_predictions/train_loss/{model_name}.pt"
        trainer.plot_losses(losses, loss_path)
        torch.save(losses, loss_path)
    
    
    # Check prediction for one example with renderer
    trainer.evaluate_model_with_renderer(test_dataloader, model_save_path, n_samples=10)
    
if __name__ == '__main__':
    main()