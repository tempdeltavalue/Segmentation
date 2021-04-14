"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import numpy as np
import os
import pandas as pd
import torch
import json

import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    load_image,
    preprocess_coco_ann
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class AEDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_size=416,
        transform=None,
    ):

        self.pre_images, self.pre_annotations = preprocess_coco_ann(csv_file)

        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(list(self.pre_images.keys()))

    def __getitem__(self, index):
        curr_ann_key = list(self.pre_images.keys())[index]
        current_img_ann = self.pre_images[curr_ann_key]
        current_ann = self.pre_annotations[curr_ann_key]
        img_url = current_img_ann["coco_url"]
        image = load_image(img_url)

        # if self.transform:
        #     augmentations = self.transform(image=image, bboxes=bboxes)
        #     image = augmentations["image"]
        #     bboxes = augmentations["bboxes"]

        targets = []

        return image, tuple(targets)


def test():
    dataset = AEDataset(
        r"C:\Users\m\Downloads\annotations_trainval2017\annotations\instances_train2017.json",
    )

    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True)

    for image, segmap in loader:
        print("image s", image[0].shape)
        tst_img = np.array(image[0])
        cv2.imshow("test image", tst_img)
        cv2.waitKey(0)

        return



if __name__ == "__main__":
    test()
