import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


import sys
sys.path.append(r'C:\\Users\\m\\Desktop\\Segmentation\\AEPytorch')


from unet_model import UNet

from utils import plot_img_and_mask
from dataset import AEDataset
from torch.utils.data import DataLoader, random_split

import cv2

def predict_img(net,
                imgs,
                device,
                true_masks,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    dataset = AEDataset(csv_file=ann_path)

    #https://stackoverflow.com/questions/56838341/dataloader-object-does-not-support-indexing

    print("before pred imgs input shape", imgs.shape)
    imgs = imgs.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()
    masks = []
    with torch.no_grad():
        print("img input shape", imgs.shape)
        output = net(imgs)   #
        print("output shape", output.shape)
        print("true_masks shape", true_masks.shape)

        print("BCEWithLogitsLoss shape", loss(output, true_masks))

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        print("probs shape", probs.shape)

        probs = probs.squeeze(0)
        print("probs after squeeze shape", probs.shape)
        print("full_img.size()", imgs.size())
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(imgs.size()[2]),
                transforms.ToTensor()
            ]
        )
        for prob in probs:
            probs = tf(prob.cpu())
            full_mask = probs.squeeze().cpu().numpy()
            masks.append(full_mask > out_threshold)

    return masks, output


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


# def get_output_filenames(args):
#     out_files = []
#
#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output
#
#     return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    # out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    print("args.model", args.model)
    print("os.path.isfile(args.model)", os.path.isfile(args.model))

    net.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded !")

    logging.info("Model loaded !")

    ann_path = r"C:\Users\m\Desktop\COCO\annotations\instances_train2017.json"
    dataset = AEDataset(csv_file=ann_path)
    print("dataset here")

    # n_val = int(len(dataset) * 0.3)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    # print("train", train)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

    for index,  (imgs, true_masks) in enumerate(train_loader):
        if index > 0:
            break

        print("start pred img shape", imgs.shape)
        masks, rare_outputs = predict_img(net=net,
                           imgs=imgs, true_masks=true_masks,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        print("finish pred masks[0] shape", masks[0].shape)

        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])
        #
        #     logging.info("Mask saved to {}".format(out_files[i]))

        logging.info("Visualizing results for image , close to continue ...")

        print(" torch.__file__",  torch.__file__)

        for index, t_img in enumerate(imgs):
            image = t_img
            image = image.permute(1, 2, 0)
            print("image shape", image.shape)
            print("mask shape", masks[index].shape)
            # print("nn.BCEWithLogitsLoss()", torch.nn.BCEWithLogitsLoss(rare_outputs, true_masks))
            plot_img_and_mask(image, masks[index])
            # plot_img_and_mask(image, true_masks[index])

