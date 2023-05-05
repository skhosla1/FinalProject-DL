import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('AAPL.csv')
data = np.array(df['Close'].values.reshape(-1))
percent_change_in_closing = (data[1:10590] - data[0:10589]) / data[0:10589]

# Normalize the data
normed_change_in_closing = np.cbrt(percent_change_in_closing) * 1.2446

# Create input/output sequences
X = []
y = []

window_size = 60 # uses this number of days' data to predict the next day
for i in range(window_size, len(data)-1):
    X.append(normed_change_in_closing[i-window_size:i])
    y.append(normed_change_in_closing[i])

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
        self.conv1 = nn.Conv1d(1, 15, kernel_size=12,stride=2)
        self.conv2 = nn.Conv1d(15,30, kernel_size=7,stride=2)
        self.conv3 = nn.Conv1d(30,60, kernel_size=5,stride=2)
        self.fc1 = nn.Linear(180, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        print(x.size())
        x = x.view(-1, 180)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = ConvNet()

# Define the loss function and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
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

    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))
    
    # if (epoch+1) % 10 == 0:
    #     print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.item()))

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    mae = nn.functional.l1_loss(y_pred, y_test.unsqueeze(1))

    y_pred_as_percent = np.array(y_pred.tolist())
    y_pred_as_percent = np.power(y_pred_as_percent / 1.2446, 3)

    y_test_as_percent = np.array(y_test.tolist())
    y_test_as_percent = np.power(y_test_as_percent / 1.2446, 3)

    percent_loss = np.mean(np.abs(y_test_as_percent - y_pred_as_percent))
    print('Test MAE: %.3f' % mae.item())
    print('Percent MAE: %.3f' % percent_loss)
    
