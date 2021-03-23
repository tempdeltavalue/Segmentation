from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import pylab

import cv2

def main():
    coco = COCO(r"C:\Users\m\Desktop\ImageSegmentation\coco\annotations\instances_minival2014.json")
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [226111])
    img_obj = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    print(img_obj)
    img = io.imread(img_obj['url'])
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()