import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from dataset import AEDataset
import sys
sys.path.append('/')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder

import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        # self.fcn = nn.Linear(in_features=1024, out_features=hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3_1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.size(0), -1)

        # x = self.fcn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = 64 #hidden_size

        self.conv3_1 = nn.ConvTranspose2d(64, 16, 3)
        self.conv3_2 = nn.ConvTranspose2d(16, 16, 3)
        self.conv3_3 = nn.ConvTranspose2d(16, 16, 3, stride=2)
        self.conv3_4 = nn.ConvTranspose2d(16, 16, 3, stride=2)
        self.conv3_5 = nn.ConvTranspose2d(16, 16, 5, stride=2)
        self.conv3_6 = nn.ConvTranspose2d(16, 1, 5, stride=2)


    def forward(self, imgs):
        x = imgs
        x = x.view(x.size(0), 64, 1, 1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.conv3_6(x)

        x = nn.Softmax2d()(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AE(object):
    def __init__(self, ann_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = AEDataset(csv_file=ann_path)

        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = AutoEncoder(hidden_size=224)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return BCE

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, y_data) in enumerate(self.train_loader):
            data = data.to(self.device, dtype=torch.float)
            y_data = y_data.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            recon_batch = self.model(data)

            loss = self.loss_function(recon_batch, y_data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    model = AutoEncoder(hidden_size=256)
    x = torch.randn(1, 3, 224, 224)

    # Let's print it
    preds = model(x)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("outputshape", preds.shape)
    print("pytorch_total_params", pytorch_total_params)

