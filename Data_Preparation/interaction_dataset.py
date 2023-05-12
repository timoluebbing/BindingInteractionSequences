import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
from Data_Preparation.data_preparation import Preprocessor


class TimeSeriesDataset(Dataset):
    def __init__(self, interaction_paths, use_distances_and_motor=True, transform=None):
        """
        Args:
            interaction_paths (dict): Labels with paths to the preprocessed csv files for each interaction
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.interaction_paths = interaction_paths
        self.use_distances_and_motor = use_distances_and_motor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        sequence, label = self.prepro.create_inout_sequence(sample)
        
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
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    ##### Dataset and DataLoader #####
    dataset = TimeSeriesDataset(interaction_paths)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    print(f"Number of samples: {len(dataset)}")
    
    example = next(iter(dataloader))
    seq, label, interaction = example
    print("\nExample sequence:")
    print(f"Sequence shape:  {seq.shape} {seq.dtype}")
    print(f"Label shape:     {label.shape} {label.dtype}")
    print(f"Interaction:     {interaction} {interaction.dtype}")
    
if __name__ == "__main__":
    main()