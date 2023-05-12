import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[:,idx,:], self.y[:,idx,:]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        print(x.shape)
        seq_len, batch_size, _ = x.size()
        print(batch_size, seq_len)
        hidden_state = torch.zeros(batch_size, hidden_size)
        cell_state = torch.zeros(batch_size, hidden_size)

        for t in range(seq_len):
            hidden_state, cell_state = self.lstm_cell(x[:, t, :], (hidden_state, cell_state))

        output = self.linear(hidden_state)
        return output

def create_dataset(dataset):
    X = dataset[:-1]
    y = dataset[1:]
    return X, y
    

# Generate dummy time series data
timeseries = torch.randn(10, 200, 5)
print(timeseries.shape)

# Create datasets with fixed windows
X_train, y_train = create_dataset(timeseries)
print(X_train.shape, y_train.shape)

# Define an LSTM model
input_size = 5
hidden_size = 10
model = LSTMModel(input_size, hidden_size)

# Set up a DataLoader
dataset = TimeSeriesDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

x, y = next(iter(dataloader))
print(x.shape, y.shape)

# Training loop
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5):
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.permute(1,0,2)
        y_batch = y_batch.permute(1,0,2)
        # Forward pass through the LSTM
        output = model(X_batch)

        # Calculate the loss
        loss = criterion(output, y_batch)

        # Zero the gradients and perform backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Update the model's parameters
        optimizer.step()

print("Finished training for 100 epochs.")