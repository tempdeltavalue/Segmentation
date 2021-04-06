import os
import cv2
import random
import json
import urllib.request
import numpy as np
import time
import glob

IMAGE_DIR = os.path.join(os.getcwd(), "Mask_RCNN_tf_fork_matterport/images")

class Utils:
    @staticmethod
    def get_image(img_path):
        # print("IMAGE_DIR", IMAGE_DIR)
        # file_names = next(os.walk(IMAGE_DIR))[2]
        # test_img_path = r'C:\Users\m\Desktop\Image Segmentation\keras_deeplab_v3_plus\imgs\image1.jpg'
        # random_path = os.path.join(IMAGE_DIR, random.choice(file_names))
        # print("Current random path", random_path)
        image = cv2.imread(img_path)
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
    def generate_coco_subset():
        ann_path = r"C:\Users\m\Downloads\annotations_trainval2017\annotations\instances_train2017.json"

        result_dict = {}

        with open(ann_path) as f:
            data = json.load(f)
            images = data["images"]
            annotations = data["annotations"]

            save_root_path = r"C:\Users\m\Desktop\COCOtestset"
            save_base_path = save_root_path + r"\new_val2017"
            saved_imgs_count = 0
            print("len images", len(images))
            for img_data in images:
                img_id = img_data["id"]
                # if img_id != 183666:
                #     continue
                print("VOVA")

                if saved_imgs_count > 100:
                    break

                for ann in annotations:
                    category_id = ann["category_id"]

                    if category_id == 1 and ann["image_id"] == img_id:
                        # segmentation_maps = ann["segmentation"]
                        # box = ann["bbox"]
                        # for seg_map in segmentation_maps:
                        #     poly = np.array(seg_map).reshape((int(len(seg_map)/2), 2))

                        url = img_data["coco_url"]
                        start_time = time.time()
                        try:
                            with urllib.request.urlopen(url) as url:
                                s = url.read()

                                arr = np.asarray(bytearray(s), dtype=np.uint8)
                                img = cv2.imdecode(arr, -1)  # 'Load it as it is'

                                img_file_name = img_data["file_name"]
                                img_save_path = os.path.join(save_base_path, img_file_name)

                                if img_file_name not in result_dict:
                                    result_dict[img_file_name] = []

                                result_dict[img_file_name].append(ann["bbox"])

                                cv2.imwrite(img_save_path, img)
                                saved_imgs_count += 1
                                print("saved_imgs_count", saved_imgs_count)
                                print("LOADING TIME", time.time() - start_time)
                                # mask = np.zeros((img.shape[0], img.shape[1]))
                                #
                                # cv2.fillConvexPoly(mask, np.int32([poly]), 1)
                                # mask = mask.astype(np.bool)
                                # out = np.zeros_like(img)
                                # out[mask] = img[mask]

                                # img = Utils.apply_mask(img, mask, [125, 125, 125])
                            # cv2.imshow('lalala', img)
                            # if cv2.waitKey() & 0xff == 27:
                            #     continue

                        except Exception as e:
                            print("Error", e)

        print("result_dict", result_dict)

        print("PRE JSON SAVED")
        with open(save_root_path + '/data.json', 'w') as fp:
            print("JSON SAVED")
            json.dump(result_dict, fp)

if __name__ == "__main__":
    Utils.generate_coco_subset()