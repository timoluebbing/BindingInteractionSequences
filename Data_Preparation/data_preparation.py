import contextlib
import pandas as pd
import torch
import numpy as np
import math
import sys

laptop_dir = "C:\\Users\\timol\\Desktop\\BindingInteractionSequences"
sys.path.append(laptop_dir)
import Data_Preparation.interaction_simulation as sim


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
    
    
    ################################################################
    # Prepare raw interaction data and save 
    # preprocessed data to concat file
    ################################################################
    def load_dataframe(self, csv_path):
        df = pd.read_csv(csv_path)
        with contextlib.suppress(Exception):
            df.drop(columns='frame', inplace=True)
        return df
    
    def load_preprocessed_dataframe(self, csv_path):
        df = self.load_dataframe(csv_path)
        return self.add_features_to_interaction_dataframe(df)
        
        
    def add_features_to_interaction_dataframe(self, df):

        df = self.normalize_coordinates_dataframe(df)  # , norm_to_center=True        
        df = self.add_distances_to_dataframe(df, normalize=False)
        df = self.convert_orientation_dataframe(df)
        
        # Reindex columns to correct input ordering
        columns = ['actor1_x', 'actor1_y', 'actor1_o_sin', 'actor1_o_cos', 'actor1_col_force_x', 'actor1_col_force_y',
                   'actor2_x', 'actor2_y', 'actor2_o_sin', 'actor2_o_cos', 'actor2_col_force_x', 'actor2_col_force_y',
                   'ball_x', 'ball_y', 'ball_o_sin', 'ball_o_cos', 'ball_col_force_x', 'ball_col_force_y',
                   'motor_fx', 'motor_fy',
                   'dis_act1_act2', 'dis_act1_ball', 'dis_act2_ball']
        df = df[columns]  # instead of dropping all the other columns
        
        return df.reindex(columns=columns)
    
    def add_distances_to_dataframe(self, df, normalize=False):
        def cal_dis(x1, y1, x2, y2):
                return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            
        # Add distances to dataframe (min max normalized to [0-1])
        df['dis_act1_act2'] = df.apply(
            lambda row: cal_dis(row['actor1_x'], row['actor1_y'], row['actor2_x'], row['actor2_y']), 
            axis=1)
        df['dis_act1_ball'] = df.apply(
            lambda row: cal_dis(row['actor1_x'], row['actor1_y'], row['ball_x'], row['ball_y']), 
            axis=1)
        df['dis_act2_ball'] = df.apply(
            lambda row: cal_dis(row['actor2_x'], row['actor2_y'], row['ball_x'], row['ball_y']), 
            axis=1)
        
        if normalize:
            pass
            # falsch, da abhängig von trial!!!
            # df['dis_act1_act2'] = (df['dis_act1_act2'] - df['dis_act1_act2'].min()) / (df['dis_act1_act2'].max() - df['dis_act1_act2'].min())
            # df['dis_act1_ball'] = (df['dis_act1_ball'] - df['dis_act1_ball'].min()) / (df['dis_act1_ball'].max() - df['dis_act1_ball'].min())
            # df['dis_act2_ball'] = (df['dis_act2_ball'] - df['dis_act2_ball'].min()) / (df['dis_act2_ball'].max() - df['dis_act2_ball'].min())

        return df
    
    def normalize_coordinates_dataframe(self, df, center=False):
        window_height = sim.HEIGHT
        window_width = sim.WIDTH
        
        center_x = window_width / 2
        center_y = window_height / 2 # account for boundaries
        
        if center:
            # Normalize coordinates to bottom center of window between actors
            df[['actor1_x', 'actor2_x', 'ball_x']] = df[['actor1_x', 'actor2_x', 'ball_x']] - center_x
            df[['actor1_y', 'actor2_y', 'ball_y']] = df[['actor1_y', 'actor2_y', 'ball_y']] - center_y # Das macht irgendwie noch keinen Sinn
        else: 
            # Normalize coordinates to [0-1] by window width and height
            df[['actor1_x', 'actor2_x', 'ball_x']] = df[['actor1_x', 'actor2_x', 'ball_x']] / window_width
            df[['actor1_y', 'actor2_y', 'ball_y']] = df[['actor1_y', 'actor2_y', 'ball_y']] / window_height
        
        return df
    
    def normalize_forces(self, df):
        # Normalize to [-1, 1] by abs highest values centered around 0 over all sequences
        def norm(x, column):
            if x == 0: # Works without this if clause, but speeds things up
                return 0.0
            elif df[column].abs().max() == - df[column].min():
                return 2 * ((x - df[column].min()) / (df[column].abs().max() - df[column].min())) - 1
            else:
                return 2 * ((x + df[column].max()) / (df[column].max() + df[column].max())) - 1
        
        columns = ['motor_fx', 'motor_fy', 
                   'actor1_col_force_x', 'actor1_col_force_y', 
                   'actor2_col_force_x', 'actor2_col_force_y',
                   'ball_col_force_x', 'ball_col_force_y',]
        
        for column in columns:
            df[column] = df[[column]].applymap(lambda x: norm(x, column))
            
        
        # df['motor_fx'] = df[['motor_fx']].applymap(lambda x: norm(x, 'motor_fx'))
        # df['motor_fy'] = df[['motor_fy']].applymap(lambda x: norm(x, 'motor_fy'))
        # df['actor1_col_force_x'] = df[['actor1_col_force_x']].applymap(lambda x: norm(x, 'actor1_col_force_x'))
        # df['actor1_col_force_y'] = df[['actor1_col_force_y']].applymap(lambda x: norm(x, 'actor1_col_force_y'))
        # df['actor2_col_force_x'] = df[['actor2_col_force_x']].applymap(lambda x: norm(x, 'actor2_col_force_x'))
        # df['actor2_col_force_y'] = df[['actor2_col_force_y']].applymap(lambda x: norm(x, 'actor2_col_force_y'))
        # df['ball_col_force_x'] = df[['ball_col_force_x']].applymap(lambda x: norm(x, 'ball_col_force_x'))
        # df['ball_col_force_x'] = df[['ball_col_force_x']].applymap(lambda x: norm(x, 'ball_col_force_x'))
        
        return df
    
    def convert_orientation_dataframe(self, df): 
        # Convert radient to sin(radient)
        df[['actor1_o_sin', 'actor2_o_sin', 'ball_o_sin']] = df[['actor1_o', 'actor2_o', 'ball_o']].applymap(math.sin)
        df[['actor1_o_cos', 'actor2_o_cos', 'ball_o_cos']] = df[['actor1_o', 'actor2_o', 'ball_o']].applymap(math.cos)

        return df
    
    def get_csv_paths(self, number_of_files, interaction='A'):
        
        return [
            f"Data_Preparation/Interactions/{interaction}/interaction_{interaction}_trial_{csv_id}.csv"
            for csv_id in range(number_of_files)
        ]
      
    def concat_csv_files(self, number_of_files, output_path, preprocess=True, interaction='A'):
        
        csv_paths = self.get_csv_paths(number_of_files, interaction=interaction)
        
        df_list = []
        for i, path in enumerate(csv_paths):
            print(f"Processing seq {i}") if i % 50 == 0 else None
            df = pd.read_csv(path) 
            
            if preprocess:
                df = self.add_features_to_interaction_dataframe(df)
            
            df.insert(0, 'video_id', i)
            df_list.append(df)
        
        df_concat = pd.concat(df_list)
        
        if preprocess:
            df_concat = self.normalize_forces(df_concat)
        
        df_concat.to_csv(output_path, index=False)
    
    ################################################################
    # Load and compile preprocessed data for lstm
    ################################################################
    def load_concat_dataframe(self, df_concat_path):
        df_concat = self.load_dataframe(df_concat_path)
        # return list of sequence dataframes
        return [
            df_concat[df_concat['video_id'] == i]
            for i in range(df_concat['video_id'].max() + 1)
        ]
    
    def dataframes_to_tensor(self, df_list):
        """ Retruns a stacked Tensor with all sequences for one interaction

        Args:
            df_list ([pd.Dataframe]): List of sequence dataframes

        Returns:
            torch.Tensor: Tensor of shape (num interactions, sequence_length, features)
        
        """        
        tensor_list = [
            self.dataframe_to_tensor(df)
            for df in df_list
        ]
        
        return torch.stack(tensor_list)
            
    
    def dataframe_to_tensor(self, df):
        
        with contextlib.suppress(Exception):
            df.drop(columns='frame', inplace=True)
        with contextlib.suppress(Exception):
            df.drop(columns='video_id', inplace=True)

        return torch.Tensor(df.values)
        
        
    def create_inout_sequence(self, input_data, n_features):
        """ Creates input and label sequences for a single interaction sequence

        Args:
            input_data (torch.Tensor): Tensor of shape (sequence_length, features)
            n_features (int)         : Number of output features

        Returns:
            tuple: Tuple of input and label sequence
        """        
        seq = input_data[:-1]
        label = input_data[1:]#

        if n_features == 18:
            return seq, label[:,:n_features]

        if n_features == 12:
            # das sollte noch über self.num_dim und self.num_features abstrahiert werden
            ls = [
                label[: , i*4 + i*2: (i+1)*4 + i*2]  
                for i in range(self.num_features)
            ]
            return seq, torch.cat(ls, dim=1)
    
    
    def get_LSTM_data_interaction(
        self, 
        path, 
        no_forces=False, 
        use_distances_and_motor=False, 
        distances_and_motor_only=False
    ):
        """ Returns interaction specific tensor data

        Args:
            path (str)                     : Path to preprocessed csv data
            use_distances (bool, optional) : Flag to return feature data with distances 
            distances_only (bool, optional): Flag to return distances only without features
            
        Returns:
            torch.Tensor: Tensor of shape (num sequences, sequence_length, features)
        """        
        sequence_list = self.load_concat_dataframe(path)
        tensor_list = self.dataframes_to_tensor(sequence_list)

        if no_forces and use_distances_and_motor:
            
            tensors = []
            for i in range(self.num_features):
                t = tensor_list[:, : , i*4 + i*2: (i+1)*4 + i*2]
                tensors.append(t)
                
            dis_motor = tensor_list[:, :, (self.num_dimensions * self.num_features) : ]
            tensors.append(dis_motor)
            
            return torch.cat(tensors, dim=2)
        
        if use_distances_and_motor:
            return tensor_list

        if distances_and_motor_only:
            return tensor_list[:, :, (self.num_dimensions * self.num_features) : ]

        return tensor_list[:, :, : (self.num_dimensions * self.num_features)]
    

def main(interaction = 'A', save_concat=False):

    path = f"Data_Preparation/Interactions/{interaction}/interaction_{interaction}_trial_0.csv"
    concat_path=f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat.csv"
    concat_raw_path=f"Data_Preparation/Interactions/Data/interaction_{interaction}_concat_raw.csv"
    
    prepro = Preprocessor(num_features=3, num_dimensions=6) # wozu zählen die distanzen?
    
    # raw_seq = prepro.load_dataframe(path)
    # print(raw_seq.shape)
    # print(raw_seq.head(20))
    
    # test_seq = prepro.load_preprocessed_dataframe(path)
    # print(test_seq.shape)
    # print(test_seq.tail(40))
    # print(test_seq.head(40))
    # print(test_seq['dis_act1_act2'].to_string())
    
    if save_concat:
        print("Concatenating csv files")
        prepro.concat_csv_files(number_of_files=300, 
                                output_path=concat_path,
                                preprocess=True, 
                                interaction=interaction)

    tensor_data = prepro.get_LSTM_data_interaction(
        concat_path, 
        no_forces=True,
        use_distances_and_motor=True)
    print(tensor_data.shape)
    # print(tensor_data[0,:40,-6:])
        
    # df = pd.DataFrame({
    #     'motor_fx': [-6,-5,-4,-3,-1,0,0,0,0,1,2,3,4],
    #     'motor_fy': [-4,-3,-1,0,0,0,0,1,2,3,4,5,6],
    # })
    
    # print(df, end='\n\n')
    # test = prepro.normalize_motor_force(df)
    # print(test)
    
    
    
if __name__ == '__main__':
    main()