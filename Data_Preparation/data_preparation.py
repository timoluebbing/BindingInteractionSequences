import pandas as pd
import torch
import numpy as np
import math

def cal_dis(x1, y1, x2, y2):
        return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def main():

    interaction = 'A'
    path = f"Data_Preparation/Interactions/interaction_{interaction}.csv"
    save_to_path = f"Data_Preparation/Interactions/interaction_{interaction}.pt"

    df = pd.read_csv(path, sep=',')
    print(df.head)
    print(df.columns)

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
    
    print(df.tail(60))

    # Pandas dataframe to torch Tensor
    tensor_data = torch.Tensor(df.values)
    print(tensor_data.shape)
    print(tensor_data.dtype)
    print(tensor_data.device)

    torch.save(tensor_data, save_to_path)
    tensor = torch.load(save_to_path)
    print(tensor.shape)

if __name__ == '__main__':
    main()