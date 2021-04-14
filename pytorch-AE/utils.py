import numpy as np
import torch
import urllib.request
import cv2
import json

def get_interpolations(args, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps+2)
            interps = []
            for a in alpha:
                interps.append(a*t2.view(1, -1) + (1 - a)*t1.view(1, -1))
            return torch.cat(interps, 0)

        if args.model == 'VAE':
            mu, logvar = model.encode(images.view(-1, 784))
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif args.model == 'AE':
            embeddings = model.encode(images.view(-1, 784))
            
        interps = []
        for i in range(0, images_per_row+1, 1):
            interp = interpolate(embeddings[i], embeddings[i+1], images_per_row-4)
            interp = interp.to(device)
            interp_dec = model.decode(interp)
            line = torch.cat((images[i].view(-1, 784), interp_dec, images[i+1].view(-1, 784)))
            interps.append(line)
        # Complete the loop and append the first image again
        interp = interpolate(embeddings[i+1], embeddings[0], images_per_row-4)
        interp = interp.to(device)
        interp_dec = model.decode(interp)
        line = torch.cat((images[i+1].view(-1, 784), interp_dec, images[0].view(-1, 784)))
        interps.append(line)

        interps = torch.cat(interps, 0).to(device)
    return interps

def load_image(url):
    with urllib.request.urlopen(url) as url:
        s = url.read()

        arr = np.asarray(bytearray(s), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img

def preprocess_coco_ann(ann_path):
    pre_images = {}
    pre_annotations = {}

    with open(ann_path) as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]

    for ann in annotations:
        image_id = ann['image_id']

        if image_id not in pre_annotations:
            pre_annotations[image_id] = []

        pre_annotations[image_id].append(ann)

        for image in images:
            if image_id in pre_images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
                break
            else:
                pre_images[image_id] = image
                break

    return pre_images, pre_annotations
