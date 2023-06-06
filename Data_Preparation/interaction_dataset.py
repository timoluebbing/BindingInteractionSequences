import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

pc_dir = "C:\\Users\\TimoLuebbing\\Desktop\\BindingInteractionSequences"
laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(pc_dir)
from Data_Preparation.data_preparation import Preprocessor


class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 interaction_paths, 
                 n_out=18, 
                 use_distances_and_motor=True, 
                 transform=None,
                 num_samples=1200):
        """
        Custom dataset class for interaction sequences. If train, validate and test flags are all False, the whole dataset
        is returned (default).
        
        Args:
            interaction_paths (dict)      : Labels with paths to the preprocessed csv files for each interaction
            n_out (int)                   : Number of lstm output features
            use_distances_and_motor (bool): Flag indicating to use additional input features
            transform (callable, optional): Optional transform to be applied on a sample.
            num_samples (int)             : Number of sequences
            split (tuple)                 : train, val, test split
            train, val, test (bool)       : Flags to select dataset purpose 
        """
        self.interaction_paths = interaction_paths
        self.n_out = n_out
        self.use_distances_and_motor = use_distances_and_motor
        self.num_samples = num_samples
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prepro = Preprocessor(num_features=3, num_dimensions=6)

        self.data_list = []
        self.interactions = np.array([])
        
        self._load_interactions()
        
        self.data = torch.cat(self.data_list, dim=0)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sequence, label = self.prepro.create_inout_sequence(sample, self.n_out)
        
        interaction = self.interactions[idx]

        if self.transform:
            sample = self.transform(sample)

        return sequence, label, interaction
    
    def _load_interactions(self):
                
        for interaction, path in self.interaction_paths.items():
                print(f"Loading interaction {interaction} from {path}")
                interaction_data = self.prepro.get_LSTM_data_interaction(path, self.use_distances_and_motor)
                print(interaction_data.shape, end='\n\n')
                
                num_sequences = len(interaction_data)
                interaction_labels = np.full(shape=(num_sequences, ), fill_value=interaction)
                
                self.interactions = np.concatenate([self.interactions, interaction_labels], axis=0)
                self.data_list.append(interaction_data)
    
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
    dataset = TimeSeriesDataset(interaction_paths)
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    
    print(f"Number of train samples:     {len(train_dataset)}")
    print(f"Number of valiation samples: {len(val_dataset)}")
    print(f"Number of test samples:      {len(test_dataset)} \n")
    
    example = next(iter(train_dataloader))
    seq, label, interaction = example
    seq = seq.permute(1,0,2)
    label = label.permute(1,0,2)
    
    print("\nExample sequence:")
    print(f"Sequence shape:  {seq.shape} {seq.dtype}")
    print(f"Label shape:     {label.shape} {label.dtype}")
    print(f"Interaction:     {interaction} {interaction.dtype}")
    
if __name__ == "__main__":
    main()