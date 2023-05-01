import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('AAPL.csv')
data = np.array(df['Close'].values.reshape(-1, 1))
percent_change_in_closing = (data[0:10588] - data[1:10589]) / data[0:10588]
# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(percent_change_in_closing)

# Create input/output sequences
X = []
y = []
for i in range(60, len(data)):
    X.append(data[i-60:i, 0])
    y.append(data[i, 0])
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
for epoch in range(50):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, 50, loss.item()))

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    mae = nn.functional.l1_loss(y_pred, y_test.unsqueeze(1))
    print('Test MAE: %.3f' % mae.item())