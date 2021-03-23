import os
import cv2
import random
import json
import urllib.request
import numpy as np

IMAGE_DIR = os.path.join(os.getcwd(), "Mask_RCNN_tf_fork_matterport/images")

class Utils:
    @staticmethod
    def get_image():
        print("IMAGE_DIR", IMAGE_DIR)
        file_names = next(os.walk(IMAGE_DIR))[2]
        test_img_path = r'C:\Users\m\Desktop\Image Segmentation\keras_deeplab_v3_plus\imgs\image1.jpg'
        random_path = os.path.join(IMAGE_DIR, random.choice(file_names))
        print("Current random path", random_path)
        image = cv2.imread(random_path)

        return image


    @staticmethod
    def divide_pascal_train_val():
        root_path = r'C:\Users\m\Desktop\Image Segmentation\TensorflowYOLACT\data'
        imgs_folder_dir = root_path + '\img'
        new_path = root_path + '\pascal_val'

        json_path = root_path + '\pascal_sbd_val.json'
        with open(json_path) as f:
            data = json.load(f)
            images = data["images"]
            for img in images:
                img_path = os.path.join(imgs_folder_dir, img["file_name"])
                new_img_path = os.path.join(new_path, img["file_name"])
                os.replace(img_path, new_img_path)

    @staticmethod
    def apply_mask(image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image


    @staticmethod
    def test_annotations():
        ann_path = r"C:\Users\m\Desktop\ImageSegmentation\coco\annotations\instances_minival2014.json"
        with open(ann_path) as f:
            data = json.load(f)
            print(data.keys())
            images = data["images"]
            annotations = data["annotations"]

            for img in images:
                img_id = img["id"]
                # print(img_id)

                for ann in annotations:
                    ann_id = ann["id"]


                    if img_id == ann_id:
                        #print(ann)
                        segmentation_maps = ann["segmentation"]
                        box = ann["bbox"]
                        category_id = ann["category_id"]
                        print(len(segmentation_maps))
                        for seg_map in segmentation_maps:
                            poly = np.array(seg_map).reshape((int(len(seg_map)/2), 2))


                        url = img["url"]
                        print(img)
                        try:
                            with urllib.request.urlopen(url) as url:
                                s = url.read()

                                arr = np.asarray(bytearray(s), dtype=np.uint8)
                                img = cv2.imdecode(arr, -1)  # 'Load it as it is'
                                mask = np.zeros((img.shape[0], img.shape[1]))

                                cv2.fillConvexPoly(mask, np.int32([poly]), 1)
                                mask = mask.astype(np.bool)
                                out = np.zeros_like(img)
                                out[mask] = img[mask]

                                # img = Utils.apply_mask(img, mask, [125, 125, 125])
                            cv2.imshow('lalala', out)
                            if cv2.waitKey() & 0xff == 27:
                                continue

                        except Exception as e:
                            print("Error", e)



if __name__ == "__main__":
    Utils.test_annotations()