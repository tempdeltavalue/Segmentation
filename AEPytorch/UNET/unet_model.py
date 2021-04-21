""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.detection_out = torch.nn.Conv2d(128, 4 + 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        det_x = self.up2(x, x3)
        x = self.up3(det_x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        detection_out = self.detection_out(det_x)

        return logits, detection_out

import torch
if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)
    x = torch.randn(1, 3, 224, 224)

    # Let's print it
    preds = model(x)
    print(preds[0].shape)
    print(preds[1].shape)
