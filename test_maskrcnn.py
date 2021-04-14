import os
import sys
import time

from general_utils import GeneralUtils

# Root directory of the project
ROOT_DIR = os.getcwd() #os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from Mask_RCNN_tf_fork_matterport.mrcnn import config
from Mask_RCNN_tf_fork_matterport.mrcnn import utils
import Mask_RCNN_tf_fork_matterport.mrcnn.model as modellib
from Mask_RCNN_tf_fork_matterport.mrcnn import visualize
import argparse


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN_tf_fork_matterport/samples/coco/"))  # To find local version
from Mask_RCNN_tf_fork_matterport.samples.coco import coco

IMAGE_DIR = os.path.join(ROOT_DIR, "Mask_RCNN_tf_fork_matterport/images")

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def prepare_MASK_RCNN(weights):
    # Directory to save logs and trained model
    config = InferenceConfig()
    config.display()

    MODEL_DIR = os.path.join(ROOT_DIR, "Mask_RCNN_tf_fork_matterport\logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "Mask_RCNN_tf_fork_matterport\mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed

    print("MASK RCNN MODEL_DIR", MODEL_DIR)
    # Directory of images to run detection on

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(weights, by_name=True)

    return model

def test_MASK_RCNN(weights, img_path):
    image = GeneralUtils.get_image(img_path)
    model = prepare_MASK_RCNN(weights)
    # Run detection
    start_time = time.time()
    results = model.detect([image], verbose=1)
    print("prediction time", time.time() - start_time)

    # Visualize results
    r = results[0]
    visualize.display_instances(image,
                                r['rois'],
                                r['masks'],
                                r['class_ids'],
                                class_names,
                                r['scores'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--img_path', type=str)

    args = parser.parse_args()

    test_MASK_RCNN(args.weights, args.img_path)
