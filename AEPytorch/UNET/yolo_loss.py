"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
from unet_model import UNet
from torch.utils.data import DataLoader

from utils import intersection_over_union
import sys
sys.path.append(r'C:\\Users\\m\\Desktop\\Segmentation\\AEPytorch')
from dataset import AEDataset

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1

    def forward(self, predictions, target):
        obj_target = target[:, 4, :, :]
        obj_pred = predictions[:, 4, :, :]

        obj = obj_target == 1
        noobj = obj_target == 0

        object_loss = self.bce(obj_pred[obj],
                               obj_target[obj])


        no_object_loss = self.bce(obj_pred[noobj],
                                  obj_target[noobj])

        target_boxes = target[:, 0:4, :, :]
        pred_boxes = predictions[:, 0:4, :, :]

        box_loss = self.mse(target_boxes, pred_boxes)
        return self.lambda_box * box_loss, self.lambda_obj * object_loss, self.lambda_noobj * no_object_loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ann_path = r"C:\Users\m\Desktop\COCO\annotations\instances_train2017.json"
    dataset = AEDataset(csv_file=ann_path)

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    for imgs, (true_masks, true_detection_anc) in train_loader:
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        true_detection_anc = true_detection_anc.to(device=device)

        masks, preds = net(imgs)

        loss = YoloLoss()
        print(loss(preds, true_detection_anc))
