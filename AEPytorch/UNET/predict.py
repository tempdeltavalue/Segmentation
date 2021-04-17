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


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    dataset = AEDataset(csv_file=ann_path)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    #https://stackoverflow.com/questions/56838341/dataloader-object-does-not-support-indexing
    imgs = None
    true_masks = None

    for imgs, true_masks in train_loader:
        imgs = imgs
        true_masks = true_masks
        continue


    imgs = imgs[0].to(device=device, dtype=torch.float32)

    with torch.no_grad():
        print("img input shape", img.shape)
        output = net(imgs[None, ...])

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size()[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


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

    ann_path = r"C:\Users\m\Downloads\annotations_trainval2017\annotations\instances_train2017.json"
    dataset = AEDataset(csv_file=ann_path)
    print("dataset here")

    # n_val = int(len(dataset) * 0.3)
    # n_train = len(dataset) - n_val
    # train, val = random_split(dataset, [n_train, n_val])
    # print("train", train)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

    for index,  (img, true_masks) in enumerate(train_loader):
        if index > 0:
            break

        print("start pred")
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        print("finish pred")

        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])
        #
        #     logging.info("Mask saved to {}".format(out_files[i]))

        logging.info("Visualizing results for image , close to continue ...")
        plot_img_and_mask(img[0], mask)
