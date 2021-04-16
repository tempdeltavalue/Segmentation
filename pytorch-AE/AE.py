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
import torchvision
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        model = torchvision.models.resnet101(pretrained=True)
        for index, param in enumerate(model.parameters()):
            param.requires_grad =  index > 280

        self.basemodel = nn.Sequential(*list(model.children())[:-2])

        # self.fcn = nn.Linear(in_features=1024, out_features=hidden_size)

    def forward(self, x):
        x = self.basemodel(x)
        return x


class PrimitiveResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride)

        self.conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=2)

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.conv1x1(x)

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        h_comp = out.shape[3] - identity.shape[3]
        w_comp = out.shape[2] - identity.shape[2]
        # without /2 round here
        identity = nn.ConstantPad2d((w_comp, 0, h_comp, 0), 0)(identity)
        out += identity
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PrimitiveResBlock(2048, 32, stride=2)
        self.conv2 = PrimitiveResBlock(32, 64, stride=2)
        self.conv3 = PrimitiveResBlock(64, 64, stride=2)
        self.conv4 = PrimitiveResBlock(64, 1, stride=2)

        # self.conv3_1 = nn.ConvTranspose2d(2048, 32, 3)
        # self.conv3_2 = nn.ConvTranspose2d(32, 32, 3)
        # self.conv3_3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        # self.conv3_4 = nn.ConvTranspose2d(32, 64, 3, stride=2)
        # self.conv3_5 = nn.ConvTranspose2d(64, 64, 3, stride=2)
        # self.conv3_6 = nn.ConvTranspose2d(64, 1, 3)


    def forward(self, x):
        # x = imgs
        # x = x.view(x.size(0), 64, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = nn.Softmax2d()(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

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

        self.model = AutoEncoder()

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
    # model = torchvision.models.resnet101(pretrained=True)
    # model = Encoder()
    model = AutoEncoder()
    # model = PrimitiveResBlock(3, 32, stride=2)
    x = torch.randn(1, 3, 224, 224)


    # model = AutoEncoder(hidden_size=256)
    # #
    # # # Let's print it
    preds = model(x)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("outputshape", preds.shape)
    print("pytorch_total_params", pytorch_total_params)

