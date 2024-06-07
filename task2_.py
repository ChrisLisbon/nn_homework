"""
Task 2. RNN

Develop RNN, GRU and LSTM to predict Usage_kWh. Dataset - http://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption.

Hyperparameters are at your discretion

Compare the quality of the MSE, RMSE and R^2 models
"""
import numpy as np
import pandas as pd
from torch import optim
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

# fetch dataset
'''steel_industry_energy_consumption = fetch_ucirepo(id=851)

# data (as pandas dataframes)
X = steel_industry_energy_consumption.data.features
y = steel_industry_energy_consumption.data.targets

encoder = LabelEncoder()
encoder.fit_transform(X['WeekStatus'].to_frame())
X['WeekStatus'] = encoder.transform(X['WeekStatus'].to_frame())

encoder = LabelEncoder()
encoder.fit_transform(X['Day_of_week'].to_frame())
X['Day_of_week'] = encoder.transform(X['Day_of_week'].to_frame())

encoder = LabelEncoder()
encoder.fit_transform(y['Load_Type'].to_frame())
y['Load_Type'] = encoder.transform(y['Load_Type'].to_frame())

dataset = X
dataset['Load_Type'] = y['Load_Type']
dataset.to_csv('kw_dataset.csv', index=False)'''

dataset = pd.read_csv('kw_dataset.csv')
X = dataset.drop(columns=['Load_Type'])
y = dataset['Load_Type'].values


temp = np.zeros((y.shape[0], 3))
trans_target = y.astype('int')
temp[np.arange(y.shape[0]).astype(int), trans_target] = 1
y = temp

X = np.expand_dims(X.to_numpy().astype(float), axis=1)
#y = np.expand_dims(y.astype(float), axis=1)

split_ratio = int(y.shape[0]*0.8)
X_train = X[:split_ratio]
X_test = X[split_ratio:]
y_train = y[:split_ratio]
y_test = y[split_ratio:]

dataset = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train))
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
print('Loader created')

# Create RNN
input_dim = 9    # input dimension
hidden_dim = 50  # hidden layer dimension
layer_dim = 1     # number of hidden layers
output_dim = 3   # output dimension
device = 'cuda'

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

epochs = 500

for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = model(train_features)

        train_loss = criterion(outputs, test_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))