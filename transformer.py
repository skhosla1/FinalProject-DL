import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_absolute_error

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(StockTransformer, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        print(pe[:, 1::2])
        print(torch.cos(position * div_term))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Load the dataset
df = pd.read_csv('AAPL.csv')

# Normalize the closing price
# scaler = MinMaxScaler()
# df['Close'] = scaler.fit_transform(df[['Close']])

# Define the dataset class
class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.percent_change_in_closing = (self.data[1:,4] - self.data[0:-1,4]) / self.data[0:-1,4]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        X = self.data[index:index+self.seq_len, :]
        y = self.percent_change_in_closing[index+self.seq_len - 1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Split the dataset into train and test sets
data = df[['Low', 'Open', 'Volume', 'High', 'Close']].values
split = int(len(data) * 0.8)
train_data = data[:split, :]
test_data = data[split:, :]

# Define the data loader
batch_size = 64
seq_len = 60
train_dataset = StockDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = StockDataset(test_data, seq_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the model
input_dim = 5
d_model = 4
nhead = 4
num_layers = 1
output_dim = 1
model = StockTransformer(input_dim, d_model, nhead, num_layers, output_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define the custom loss function
def custom_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Train the model with early stopping
num_epochs = 50
best_loss = float('inf')
patience = 20
counter = 0
for epoch in range(num_epochs):
    train_loss = 0
    train_batch_count = 0
    test_batch_count = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch.transpose(0, 1))
        loss = custom_loss(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_batch_count += 1
        train_loss += loss.item()

    # Evaluate the model
    with torch.no_grad():
        test_loss = 0
        for X_test, y_test in test_loader:
            y_pred = model(X_test.transpose(0, 1))
            test_loss += custom_loss(y_pred.squeeze(), y_test)
            test_batch_count += 1

        # Update the learning rate scheduler
        scheduler.step(test_loss)

        # Check if the current test loss is the best so far
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            counter += 1

        # Check if early stopping should be applied
        if counter >= patience:
            print('Early stopping after epoch {}'.format(epoch))
            break

        # if (epoch+1) % 10 == 0:
        #     print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, test_loss))
        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss/train_batch_count, test_loss/test_batch_count))

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Make predictions on the test set
with torch.no_grad():
    y_preds = []
    for X_test, y_test in test_loader:
        y_pred = model(X_test.transpose(0, 1))
        y_preds.append(y_pred)
    y_preds = torch.cat(y_preds)
    y_true = torch.tensor(test_data[seq_len:, 3], dtype=torch.float32)
    y_preds = scaler.inverse_transform(y_preds.numpy().reshape(1, -1))
    y_true = scaler.inverse_transform(y_true.numpy().reshape(1, -1))

# Print the test MAE
mae = mean_absolute_error(y_true, y_preds)
print('Test MAE: {:.4f}'.format(mae))