import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('AAPL.csv')
data = np.array(df['Close'].values.reshape(-1, 1))
percent_change_in_closing = (data[1:10590] - data[0:10589]) / data[0:10589]

# Create input/output sequences
X = []
y = []

window_size = 60 # uses this number of days' data to predict the next day
for i in range(window_size, len(data)-1):
    X.append(percent_change_in_closing[i-window_size:i, 0])
    y.append(percent_change_in_closing[i, 0])

X = torch.tensor(np.array(X)).float()
y = torch.tensor(y).float()

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Define the model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define the optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Tokenize the input data
X_train_encodings = tokenizer([str(x) for x in X_train.numpy().tolist()], truncation=True, padding=True)
X_test_encodings = tokenizer([str(x) for x in X_test.numpy().tolist()], truncation=True, padding=True)

# Convert the input data to PyTorch tensors
X_train_tensors = torch.tensor(X_train_encodings['input_ids'])
y_train_tensors = torch.tensor([[label, 1-label] for label in y_train.tolist()])
X_test_tensors = torch.tensor(X_test_encodings['input_ids'])
y_test_tensors = torch.tensor([[label, 1-label] for label in y_test.tolist()])

# Train the model
model.train()
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensors), batch_size):
        batch_X = X_train_tensors[i:i+batch_size]
        batch_y = y_train_tensors[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X, labels=batch_y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print('Epoch {} loss: {:.3f}'.format(epoch + 1, loss.item()))

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensors)
    predictions = outputs.logits.squeeze().tolist()
    mae = sum(abs(predictions[i] - y_test[i].item()) for i in range(len(y_test))) / len(y_test)
    print('Test MAE: {:.3f}'.format(mae))
