"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import numpy as np
import os
import pandas as pd
import torch
import json
import glob
import sys


import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

#sys.path.append(r"C:\Users\m\Desktop\Segmentation")
from general_utils import GeneralUtils

ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A

transform = A.Compose([
    A.Resize(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

class AEDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_size=416
    ):

        self.pre_images, self.pre_annotations = GeneralUtils.preprocess_coco_ann(csv_file)

        self.image_paths = glob.glob(os.path.join(r"C:\Users\m\Desktop\COCOhumanTrainSubset", "*"))

        self.image_size = image_size
        self.transform = transform

        self.train_loader = torch.utils.data.DataLoader(self,
                                                        batch_size=8,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self,
                                                       batch_size=8,
                                                       shuffle=True)

        self.prev_img = None
        self.prev_mask = None

    def __len__(self):
        return len(self.image_paths)

        #return len(list(self.pre_images.keys()))

    def __getitem__(self, index):
        #curr_ann_key = list(self.pre_images.keys())[index]
        path = self.image_paths[index]
        curr_ann_key = int(path.split("\\")[-1].split(".")[0])
        current_anns = self.pre_annotations[curr_ann_key]

        # current_img_ann = self.pre_images[curr_ann_key]
        # img_url = current_img_ann["coco_url"]

        image = cv2.imread(path)  #GeneralUtils.load_image(img_url)

        if len(image.shape) < 3:
            return self.prev_img, self.prev_mask

        masks = GeneralUtils.generate_map(image, current_anns)

        # merge all masks !
        global_mask = np.sum(np.array(masks), axis=0)
        # # !!!

        augmentations = transform(image=image, mask=global_mask)
        image = augmentations["image"]

        image = np.moveaxis(image, -1, 0)

        global_mask = augmentations["mask"]

        # HZ how add separate transform for mask
        # # just for training (breaks viz)
        global_mask.resize((1, 63, 63),
                           refcheck=False)

        self.prev_img = image
        self.prev_mask = global_mask

        # print("mask resized shape", global_mask.shape)
        # # !!!

        return image, global_mask


def test():
    dataset = AEDataset(
        r"C:\Users\m\Downloads\annotations_trainval2017\annotations\instances_train2017.json",
    )

    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True)

    for img, mask in loader:
        # print("img.shape", img.shape)
        # print("mask.shape", mask.shape)
        #
        # img = np.array(img[0])

        continue

        try:
            copy_img = GeneralUtils.apply_mask(img,
                                               mask,
                                               [125, 125, 125])  # here
        except Exception as e:
            print(e)
            continue

        cv2.imshow('lalala', copy_img)

        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            continue

        return


if __name__ == "__main__":
    test()
