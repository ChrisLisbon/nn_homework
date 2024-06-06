import torch
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = torchvision.datasets.MNIST('./data',
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)

train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

train_set_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True)
test_set_loader = torch.utils.data.DataLoader(val_set,
                                              batch_size=100,
                                              shuffle=True)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Generator(nn.Module):
    def __init__(self, latent_dims):
        super(Generator, self).__init__()

        self.model = nn.Sequential(nn.Linear(latent_dims, 128),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(128, 256),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(256, 512),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 1024),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(1024, int(np.prod((1, 28, 28)))),
                                   nn.Tanh()
                                   )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *(1, 28, 28))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod((1, 28, 28))), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GAN(nn.Module):
    def __init__(self, latent_dims):
        super(GAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator(latent_dims)

    def forward(self, x):
        z = self.discriminator(x)
        return self.generator(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


autoencoder = VariationalAutoencoder(15).to(device)
generator = Generator(15).to(device)
discriminator = Discriminator().to(device)

opt = torch.optim.Adam(autoencoder.parameters())
optimizer_G = torch.optim.Adam(generator.parameters())
#optimizer_D = torch.optim.Adam(discriminator.parameters())

triplet_loss = nn.TripletMarginLoss()
adversarial_loss = torch.nn.BCELoss()
l1loss = nn.L1Loss()

losses_list = []

epochs = 600
for epoch in range(epochs):
    losses = []
    g_losses = []
    for x, y in train_set_loader:
        x = x.to(device)

        '''valid = Tensor(x.size(0), 1).fill_(1.0).to(device)
        fake = Tensor(x.size(0), 1).fill_(0.0).to(device)'''

        # Train the generator
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (100, 15)))).to(device)
        gen_imgs = generator(z)
        gen_loss = l1loss(x, gen_imgs)
        gen_loss.backward()
        optimizer_G.step()

        # Train the autoencoder
        opt.zero_grad()
        x_enc = autoencoder(x)

        loss = triplet_loss(x_enc, x, gen_imgs.detach())
        losses.append(loss.item())
        g_losses.append(gen_loss.item())
        loss.backward()
        opt.step()
        
    print(f'Epoch {epoch}, loss={np.mean(losses)}, gloss={np.mean(g_losses)}')
    #print(f'Epoch {epoch}, loss={np.mean(losses)}')
    losses_list.append(np.mean(losses))

torch.save(autoencoder.state_dict(), 'VAE_GAN15_600.pt')

plt.plot(np.arange(len(losses_list)), losses_list)
plt.show()

test_losses = []
for x, y in test_set_loader:
    x = x.to(device)
    opt.zero_grad()
    x_hat = autoencoder(x)
    output = x_hat.detach().cpu().numpy()
    target = x.detach().cpu().numpy()
    '''plt.imshow(output[0][0])
    plt.show()'''
    loss = ((x - x_hat) ** 2).sum()
    test_losses.append(loss.item())
print(np.mean(test_losses))
