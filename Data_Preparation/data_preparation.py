import pandas as pd
import torch
import numpy as np
import math


class Preprocessor():
    
    def __init__(
        self,
        num_features,
        num_dimensions,
        ):
        
        self.num_features = num_features
        self.num_dimensions = num_dimensions
        
        self.nxm = False
        # ...
    
    def set_num_features(self, num_features):
        self.num_features = num_features
    
    def set_num_dimensions(self, num_dimensions):
        self.num_dimensions = num_dimensions
        
    def compile_data_csv_to_pt(self, csv_path, pt_path):
        
        input_dataframe = pd.read_csv(csv_path)
        input_dataframe.drop(columns='frame', inplace=True)
        
        self.dataframe = self.add_features_to_interaction_dataframe(input_dataframe)
        
        # Update the number of features
        self.num_features = len(self.dataframe.columns)
        
        # Pandas dataframe to torch Tensor
        tensor_data = torch.Tensor(self.dataframe.values)
        
        print(f"Tensor data shape: {tensor_data.shape}")
        print(f"Tensor data dtype: {tensor_data.dtype}")
        print(f"Tensor data device: {tensor_data.device}")

        torch.save(tensor_data, pt_path)
        
    
    def add_features_to_interaction_dataframe(self, df):
        
        def cal_dis(x1, y1, x2, y2):
                return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            
        # Add distances to dataframe
        df['dis_act1_act2'] = df.apply(
            lambda row: cal_dis(row['actor1_x'], row['actor1_y'], row['actor2_x'], row['actor2_y']), 
            axis=1)
        df['dis_act1_ball'] = df.apply(
            lambda row: cal_dis(row['actor1_x'], row['actor1_y'], row['ball_x'], row['ball_y']), 
            axis=1)
        df['dis_act2_ball'] = df.apply(
            lambda row: cal_dis(row['actor2_x'], row['actor2_y'], row['ball_x'], row['ball_y']), 
            axis=1)

        # Convert radient to sin(radient)
        df[['actor1_o', 'actor2_o', 'ball_o']] = df[['actor1_o', 'actor2_o', 'ball_o']].applymap(math.sin)
        
        return df
    
    def std_scale_data(self, input_data, scale_factor):
        # Apply normalization to input data
        normed = torch.norm(input_data, dim=2) # funk nicht, input data hat nicht das richtige Format
        
        scale_factor = 1/(np.sqrt(scale_factor) * normed.std())
        scale_mat = torch.Tensor([[scale_factor, 0, 0], 
                                  [0, scale_factor, 0], 
                                  [0, 0, scale_factor]])
        scaled = torch.matmul(input_data, scale_mat)
        
        print(f'Scaled data by factor {scale_factor}')
        print(f'New minimum: {torch.min(scaled)}')
        print(f'New maximum: {torch.max(scaled)}')
        return scaled
    
    def add_noise(self, input_data, noise_factor):
        noise = noise_factor * (torch.rand(input_data.shape) - 0.5)
        return input_data + noise


    # Das ist doch für active tuning oder?
    # tw steht für train window. das ist aber nicht gleich batchsize oder
    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append([train_seq ,train_label])
        return inout_seq

    """
        Get LSTM data for interaction sequence. 
    """
    def get_LSTM_data_interaction(
        self, 
        path, 
        frame_samples, 
        num_test_data, 
        train_window, 
        noise=None
    ):

        visual_input = torch.load(path)  

        print(f"====================\n After load:\n {visual_input.size} \n ====================")

        if noise is not None: 
            visual_input = self.add_noise(visual_input, noise) 

            
        visual_input = visual_input.reshape(
            1, 
            frame_samples, 
            (self._num_dimensions) *self.num_features
        )
        
        print(f"====================\n After reshape\n{visual_input.size} \n ====================")
        
        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        return train_inout_seq, train_data, test_data


def main():

    interaction = 'B'
    path = f"Data_Preparation/Interactions/interaction_{interaction}.csv"
    save_to_path = f"Data_Preparation/Interactions/interaction_{interaction}.pt"
    
    prepro = Preprocessor(num_features=12, num_dimensions=4) # wozu zählen die distanzen?
    
    prepro.compile_data_csv_to_pt(path, save_to_path)    
    
    print(prepro.dataframe.shape)
    print(prepro.dataframe.tail(30))
    
    tensor = torch.load(save_to_path)
    print(tensor.shape)
    
    scaled = prepro.std_scale_data(tensor, scale_factor=1)
    print(scaled[1,:])

if __name__ == '__main__':
    main()