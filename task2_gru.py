import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRU, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = layer_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out


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
X_train = X[:split_ratio]
X_test = X[split_ratio:]
y_train = y[:split_ratio]
y_test = y[split_ratio:]

dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
print('Loader created')


input_dim = 9    # input dimension
hidden_dim = 50  # hidden layer dimension
layer_dim = 1     # number of hidden layers
output_dim = 1   # output dimension
device = 'cuda'

model = GRU(input_dim, hidden_dim, layer_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
torch.save(model.state_dict(), 'gru_300_norm.pt')
plt.plot(np.arange(epochs), losses)
plt.title('Convergence GRU')
plt.ylabel('L1loss')
plt.xlabel('Epochs')
plt.show()

model = GRU(input_dim, hidden_dim, layer_dim, output_dim).to(device)
model.load_state_dict(torch.load('gru_300_norm.pt'))
model.eval()

test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
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