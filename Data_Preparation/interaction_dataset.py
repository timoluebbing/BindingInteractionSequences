import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
from Data_Preparation.data_preparation import Preprocessor


class TimeSeriesDataset(Dataset):
    def __init__(self, interaction_paths, use_distances=False, transform=None):
        """
        Args:
            interaction_paths (dict): Paths to the preprocessed csv files for each interaction
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.prepro = Preprocessor(num_features=3, num_dimensions=4)

        data_list = []
        self.interactions = np.array([])
        
        for interaction, path in interaction_paths.items():
            print(f"Loading interaction {interaction} from {path}")
            interaction_data = self.prepro.get_LSTM_data_interaction(path, use_distances)
            print(interaction_data.shape)
            num_sequences = len(interaction_data)
            interaction_labels = np.full(shape=(num_sequences, ), fill_value=interaction)
            
            self.interactions = np.concatenate([self.interactions, interaction_labels], axis=0)
            data_list.append(interaction_data)
        
        self.data = torch.cat(data_list, dim=0)
        
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
    
    
def main():
    
    interactions = ['A', 'B', 'C', 'D']
    interactions_num = [0, 1, 2, 3]
    
    paths = [
        f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
        for interaction in interactions
    ]
    interaction_paths = dict(zip(interactions_num, paths))
    print(interaction_paths)
    
    dataset = TimeSeriesDataset(interaction_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(len(dataset))
    
    example = next(iter(dataloader))
    
    seq, label, interaction = example
    print(seq.shape)
    print(label.shape)
    print(interaction)
    
if __name__ == "__main__":
    main()