import torch 
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, random_split

import sys
pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)      
# Before run: replace ... with current directory path

from Data_Preparation.interaction_dataset import TimeSeriesDataset
from CoreLSTM.test_core_lstm import LSTM_Tester


def main(render=True):
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    
    ##### Dataset and DataLoader #####
    batch_size = 180
    timesteps = 126
    seed = 2023
    no_forces = True
    no_forces_out = False
    n_out = 12 if (no_forces or no_forces_out) else 18
    
    dataset = TimeSeriesDataset(
        interaction_paths, 
        timesteps=timesteps,
        no_forces=no_forces,
        n_out=n_out,
        use_distances_and_motor=True)
    generator = torch.Generator().manual_seed(seed)
    split = [0.7, 0.15, 0.15]
    split = [0.6, 0.3, 0.1]
    
    
    train_dataset, _, test_dataset = random_split(dataset, split, generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
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
    no_forces_best2 = 'core_lstm_4_3_5_360_MSELoss()_0.0001_0_240_3000_lnorm_tfs121_tfd_nf_ts121'
    no_forces_out_best = 'core_lstm_6_3_5_360_MSELoss()_0.0001_0_180_2500_lnorm_tfs200_nfo'
    
    model_name = no_forces_best2
    model_save_path = f'CoreLSTM/models/{model_name}.pt'
    
    mse_loss = nn.MSELoss()
    criterion = mse_loss
    
    tester = LSTM_Tester(
        loss_function=criterion,
        batch_size=batch_size,
        hidden_num=hidden_num,
        layer_norm=layer_norm,
        timesteps=timesteps,
        num_dim=n_dim,
        num_feat=n_features,
        num_independent_feat=n_independent,
        num_interactions=n_interactions,
        num_output=n_out,
        model_save_path=model_save_path
    )

    print("\nTest dataset:\n")
    print(f"Model in evaluation mode: {not tester.model.training}")
    _ = tester.evaluate(test_dataloader)
    total_loss, losses, obj_losses, type_losses = tester.evaluate_detailed(test_dataloader)
    print(f"\n Evaluation: Total loss of {total_loss:4f} - Sum step losses of {sum(losses)}")
    
    print(f"{total_loss:.8f} * timesteps = {total_loss * (timesteps-1)}")
    print(f"{sum(losses):.8f} * n_obj     = {sum(losses) * n_features}")
    print(f"{sum(losses):.8f} * n_types   = {sum(losses) * (n_out/n_dim)}")
    print(np.sum(obj_losses, axis=None))
    print(np.sum(type_losses, axis=None))
    
    test_loss_path = f"CoreLSTM/testing_predictions/test_loss/{model_name}_steps"
    obj_loss_path = f"CoreLSTM/testing_predictions/test_loss/{model_name}_obj"
    type_loss_path = f"CoreLSTM/testing_predictions/test_loss/{model_name}_type"
    tester.plot_losses_steps(losses, test_loss_path)
    tester.plot_losses_objects(obj_losses, obj_loss_path)
    tester.plot_losses_types(type_losses, type_loss_path)

    if render:
        # Check prediction for one example with renderer
        tester.evaluate_model_with_renderer(
            # train_dataloader, 
            test_dataloader,
            n_samples=5
        )
    
if __name__ == '__main__':
    main()