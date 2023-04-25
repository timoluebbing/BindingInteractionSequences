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
        self.use_distances = False
        # ...
    
    def set_num_features(self, num_features):
        self.num_features = num_features
    
    def set_num_dimensions(self, num_dimensions):
        self.num_dimensions = num_dimensions
    
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
        df[['actor1_o_sin', 'actor2_o_sin', 'ball_o_sin']] = df[['actor1_o', 'actor2_o', 'ball_o']].applymap(math.sin)
        df[['actor1_o_cos', 'actor2_o_cos', 'ball_o_cos']] = df[['actor1_o', 'actor2_o', 'ball_o']].applymap(math.cos)
        
        df.drop(columns=['actor1_o', 'actor2_o', 'ball_o'], inplace=True)
        
        # Reindex columns to correct input ordering
        columns = ['actor1_x', 'actor1_y', 'actor1_o_sin', 'actor1_o_cos', 
                   'actor2_x', 'actor2_y', 'actor2_o_sin', 'actor2_o_cos',
                   'ball_x', 'ball_y', 'ball_o_sin', 'ball_o_cos',
                   'dis_act1_act2', 'dis_act1_ball', 'dis_act2_ball']
        df = df.reindex(columns=columns)
        
        return df
    
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
    
    def get_csv_paths(self, number_of_files, interaction='A'):
        
        return [
            f"Data_Preparation/Interactions/C/interaction_{interaction}_trial_{csv_id}_temp.csv"
            for csv_id in range(number_of_files)
        ]
      
    def concat_csv_files(self, number_of_files, output_path, interaction='A'):
        
        csv_paths = self.get_csv_paths(number_of_files, interaction=interaction)
        
        df_list = []
        for i, path in enumerate(csv_paths):
            df = pd.read_csv(path)
            df.insert(0, 'video_id', i)
            df_list.append(df)
        
        df_concat = pd.concat(df_list)
        df_concat.to_csv(output_path, index=False)
    
    
    def compile_csv_concat_to_pt(self, csv_path, output_path, interaction='A'):
        dataframe = pd.read_csv(csv_path)
        # TODO: finish function
    
    def reshape_tensor_data(self, input_data):
        pass
        # Reshape tensor data to (num_frames, num_features, num_dimensions)
        # TODO: finish function
    
    # TODO: adapt to my data or change structure of my data
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

    # TODO: adapt to my data or change structure of my data
    # tw steht für train window. das ist aber nicht gleich batchsize oder, nein :)
    # Liste aus train sequences zum Beispiel: [0 - 9] mit label 10, dann [1 - 10] mit label 11 usw
    ### windows brauchen wir nicht, mal schauen
    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append([train_seq ,train_label])
        return inout_seq

    def get_LSTM_data_interaction(self, path, tw, distances=False):
        
        self.use_distances = distances
        
        visual_input = torch.load(path)
        
        if self.use_distances:
            # Einfach returnen weil die Data ja zur Zeit noch flattened sind
            return self.create_inout_sequences(visual_input, tw)
            # visual_input = visual_input.reshape(
            #     1, 
            #     self._frame_samples, 
            #     self.num_dimensions * self.num_features
            # )
        without_dis = visual_input[:, : (self.num_dimensions * self.num_features)]
        
        return self.create_inout_sequences(without_dis, tw)
    
    """
        Get LSTM data for interaction sequence. 
    """
    # TODO: adapt to my data or change structure of my data
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
            
        visual_input = visual_input.reshape(
            1, 
            frame_samples, 
            (self._num_dimensions) *self.num_features
        )
        
        print(f"====================\n After reshape\n{visual_input.size} \n ====================")
        
        # Beim dancer ändeers sich die motion nicht, also daten einfach aufgeteilt
        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        return train_inout_seq, train_data, test_data

def main(interaction = 'C'):

    path = f"Data_Preparation/Interactions/interaction_{interaction}.csv"
    save_to_path = f"Data_Preparation/Interactions/interaction_{interaction}.pt"
    
    prepro = Preprocessor(num_features=3, num_dimensions=4) # wozu zählen die distanzen?
    
    prepro.compile_data_csv_to_pt(path, save_to_path)    
    
    print(prepro.dataframe.shape)
    print(prepro.dataframe.tail(30))
    
    tensor = torch.load(save_to_path)
    print(tensor.shape)
    print(tensor[:2, :12])
    
    # scaled = prepro.std_scale_data(tensor, scale_factor=1)
    # print(scaled[1,:])
    
    output_path=f"Data_Preparation/Interactions/interaction_{interaction}_concat.csv"
    # prepro.concat_csv_files(number_of_files=5, output_path=output_path, interaction=interaction)

if __name__ == '__main__':
    main()