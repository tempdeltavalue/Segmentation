"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
from unet_model import UNet

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target):
        obj_target = target[:, 4, :, :]
        obj_pred = predictions[:, 4, :, :]

        obj = obj_target == 1
        noobj = obj_target == 0

        object_loss = self.bce(self.sigmoid(obj_pred[obj]), self.sigmoid(obj_target[obj]))


        no_object_loss = self.bce(obj_pred[noobj], obj_target[noobj])

        target_boxes = target[:, 0:4, :, :]
        pred_boxes = predictions[:, 0:4, :, :]

        box_loss = self.mse(target_boxes, pred_boxes)

        print("object_loss", object_loss)
        print("no_object_loss", no_object_loss)
        print("box_loss", box_loss)

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
        )

if __name__ == "__main__":
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    x = torch.rand((1, 3, 224, 224))
    preds = net(x)
    print("preds s", preds[1].shape)

    loss = YoloLoss()
    print(loss(preds[1], torch.rand([1, 5, 56, 56])))

