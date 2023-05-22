import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
from Data_Preparation.data_preparation import Preprocessor


class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 interaction_paths, 
                 n_out=18, 
                 use_distances_and_motor=True, 
                 transform=None,
                 num_samples=1200,
                 split=(0.7, 0.15, 0.15),
                 train=False,
                 validate=False,
                 test=False,
                 seed=0):
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
        self.split = split
        self.train, self.validate, self.test = train, validate, test
        self.seed = seed
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
        
        train_percent, val_percent, test_percent = self.split
        samples_per_interaction = int(self.num_samples / len(self.interaction_paths))
        
        for interaction, path in self.interaction_paths.items():
                print(f"Loading interaction {interaction} from {path}")
                interaction_data = self.prepro.get_LSTM_data_interaction(path, self.use_distances_and_motor)
                
                # Das ganze hier kann auch noch in der get_LSTM_data_interaction 
                # function versteckt werden...
                train_end = int(samples_per_interaction*train_percent)
                val_end = int(samples_per_interaction - (samples_per_interaction*test_percent))
                
                if self.train:
                    interaction_data = interaction_data[:train_end]
                elif self.validate:
                    interaction_data = interaction_data[train_end:val_end]
                elif self.test:
                    interaction_data = interaction_data[val_end:]
                    
                print(interaction_data.shape, end='\n\n')
                
                num_sequences = len(interaction_data)
                interaction_labels = np.full(shape=(num_sequences, ), fill_value=interaction)
                
                self.interactions = np.concatenate([self.interactions, interaction_labels], axis=0)
                self.data_list.append(interaction_data)
    
def main():
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    ##### Dataset and DataLoader #####
    train_dataset = TimeSeriesDataset(interaction_paths, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    val_dataset = TimeSeriesDataset(interaction_paths, validate=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    test_dataset = TimeSeriesDataset(interaction_paths, test=True)
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