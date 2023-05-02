import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('AAPL.csv')
data = np.array(df['Close'].values.reshape(-1, 1))
percent_change_in_closing = (data[1:10590] - data[0:10589]) / data[0:10589]
# Normalize the data
# scaler = MinMaxScaler()
# data_scaled_for_input = scaler.fit_transform(percent_change_in_closing)

# Create input/output sequences
X = []
y = []

window_size = 60 # uses this number of days' data to predict the next day
for i in range(window_size, len(data)-1):
    X.append(percent_change_in_closing[i-window_size:i, 0])
    y.append(percent_change_in_closing[i, 0])

X = torch.tensor(X).float()
y = torch.tensor(y).float()

# Split the data into training and testing sets
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Define the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 28, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 28)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = ConvNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    indices = np.arange(0,len(X_train))
    rng = np.random.default_rng()
    rng.shuffle(indices, 0)
    indices = torch.tensor(indices,dtype=torch.int64)
    ones_shape = torch.ones(60,1)
    prod = ones_shape * indices
    prod = torch.transpose(prod,0,1)
    indices_for_inputs = torch.tensor(prod,dtype=torch.int64)
    train_inputs = torch.gather(X_train, 0, indices_for_inputs)
    train_labels = torch.gather(y_train, 0, indices)

    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    mae = nn.functional.l1_loss(y_pred, y_test.unsqueeze(1))
    print('Test MAE: %.3f' % mae.item())