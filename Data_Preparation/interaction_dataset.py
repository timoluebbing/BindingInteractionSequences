import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
from Data_Preparation.data_preparation import Preprocessor


class TimeSeriesDataset(Dataset):
    def __init__(self, interaction_paths, transform=None):
        """
        Args:
            interaction_paths (dict): Paths to the preprocessed csv files for each interaction
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.prepro = Preprocessor(num_features=3, num_dimensions=4)
        
        self.data_dict = {}
        for interaction, path in interaction_paths.items():
            self.data_dict[interaction] = self.prepro.get_LSTM_data_interaction(path)
        
        self.data = []
        for interaction, data in self.data_dict.items():
            pass
            #self.data.
            # WIE GEHT DAS? Wahrscheinlich irgendwo früher oder
        
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
    
    # dataset = TimeSeriesDataset(path=paths)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # print(len(dataset))
    
    # example = next(iter(dataloader))
    
    # seq, label, interaction = example
    
    print(interaction_paths)
    
if __name__ == "__main__":
    main()