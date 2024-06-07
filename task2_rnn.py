import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


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

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


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
y['Load_Type'] = encoder.transform(y['Load_Type'].to_frame())'''


dataset = pd.read_csv('kw_dataset.csv')

for column in dataset.columns:
    scaler = MinMaxScaler()
    scaler.fit_transform(dataset[column].to_frame())
    dataset[column] = scaler.transform(dataset[column].to_frame())


'''plt.plot(dataset['Usage_kWh'][:5000])
plt.show()'''

X = dataset.drop(columns=['Usage_kWh'])
y = dataset['Usage_kWh'].values

X = np.expand_dims(X.to_numpy().astype(float), axis=1)
y = np.expand_dims(y.astype(float), axis=1)
split_ratio = int(y.shape[0]*0.8)
X_train = X[:split_ratio].astype(float)
X_test = X[split_ratio:].astype(float)
y_train = y[:split_ratio].astype(float)
y_test = y[split_ratio:].astype(float)

dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
print('Loader created')

# Create RNN
input_dim = 9    # input dimension
hidden_dim = 50  # hidden layer dimension
layer_dim = 1     # number of hidden layers
output_dim = 1   # output dimension
device = 'cuda'

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.L1Loss()

epochs = 300
losses = []
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
    losses.append(loss)
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
torch.save(model.state_dict(), 'rnn_300_norm.pt')
plt.plot(np.arange(epochs), losses)
plt.title('Convergence RNN')
plt.ylabel('L1loss')
plt.xlabel('Epochs')
plt.show()


model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
model.load_state_dict(torch.load('rnn_300_norm.pt'))
model.eval()

test_dataset = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test))
test_set_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
print('Loader created')

r2_losses = []
mse_losses = []
rmse_losses = []
for x, y in test_set_loader:
    x = x.to(device)
    x_hat = model(x)
    output = x_hat.detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    r2_test = r2_score(target, output)
    mse_test = mean_squared_error(target, output)
    rmse_test = root_mean_squared_error(target, output)

    r2_losses.append(r2_test)
    mse_losses.append(mse_test)
    rmse_losses.append(rmse_test)

print(np.nanmean(r2_losses))
print(np.nanmean(mse_losses))
print(np.nanmean(rmse_losses))