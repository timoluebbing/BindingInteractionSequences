import sys
import numpy as np
import torch
from torch.utils.data import Dataset


laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
import Data_Preparation.data_preparation as data_prep

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (list or ndarray): The time series data.
            labels (list or ndarray): The corresponding labels for each time series.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        label = self.labels[idx]
        interaction = self.interactions[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label, interaction
    
    
def main():
    
    preprocessor = data_prep.Preprocessor()
    
    
if __name__ == "__main__":
    main()