import os
import cv2
import random
import json
import urllib.request
import numpy as np
import time
import glob

IMAGE_DIR = os.path.join(os.getcwd(), "Mask_RCNN_tf_fork_matterport/images")


class GeneralUtils:
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
    def preprocess_coco_ann(ann_path):
        pre_images = {}
        pre_annotations = {}

        with open(ann_path) as f:
            data = json.load(f)

            images = data["images"]
            annotations = data["annotations"]

        for ann in annotations:
            category_id = ann['category_id']
            area = ann['area']

            if type(ann["segmentation"]) == dict:
                continue

            if category_id != 1 or area < 3000:  # medium-big and big humans
                continue

            image_id = ann['image_id']

            if image_id not in pre_annotations:
                pre_annotations[image_id] = []

            pre_annotations[image_id].append(ann)

        for image in images:
            image_id = image['id']

            if image_id in pre_images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            elif image_id in pre_annotations:
                pre_images[image_id] = image

        return pre_images, pre_annotations

    @staticmethod
    def load_image(url):
        with urllib.request.urlopen(url) as url:
            s = url.read()

            arr = np.asarray(bytearray(s), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
            return img

    @staticmethod
    def generate_map(img, current_anns):
        masks = []
        for cur_ann in current_anns:
            segmentation = cur_ann["segmentation"]

            for seg_map in segmentation:
                len_seg_map = len(seg_map)

                if len_seg_map < 10:
                    continue

                poly = np.array(seg_map).reshape((int(len_seg_map / 2), 2))

                mask = np.zeros((img.shape[0], img.shape[1]))
                cv2.fillConvexPoly(mask, np.int32([poly]), 1)
                mask = mask.astype(np.bool)

                out = np.zeros_like(img)
                out[mask] = img[mask]
                masks.append(mask)

        return masks

    @staticmethod
    def generate_coco_subset():
        max_count = 1000
        counter = 0
        ann_path = r"C:\Users\m\Downloads\annotations_trainval2017\annotations\instances_train2017.json"

        pre_images, pre_annotations = GeneralUtils.preprocess_coco_ann(ann_path)

        is_skip = False
        is_quit = False
        paths = glob.glob(os.path.join(r"C:\Users\m\Desktop\COCOhumanTrainSubset", "*"))
        for path in paths:
            curr_ann_key = int(path.split("\\")[-1].split(".")[0])
        # for curr_ann_key in list(pre_images.keys()):
            if is_quit:
                break

            if is_skip:
                is_skip = False
                continue

            # current_img_ann = pre_images[curr_ann_key]
            # img_url = current_img_ann["coco_url"]
            # file_name = str(current_img_ann["id"]) + ".jpg"

            current_anns = pre_annotations[curr_ann_key]
            img = cv2.imread(path) #GeneralUtils.load_image(img_url)
            # save_path = os.path.join(r"C:\Users\m\Desktop\COCOhumanTrainSubset", file_name)
            # cv2.imwrite(save_path, img)
            # print(save_path)
            # counter += 1

            # if max_count == counter:
            #     return

            # continue

            masks = GeneralUtils.generate_map(img, current_anns)

            # merge all masks !
            global_mask = np.sum(np.array(masks), axis=0)
            masks = [global_mask]
            # # !!!

            for mask in masks:
                try:
                    copy_img = GeneralUtils.apply_mask(img, mask, [125, 125, 125]) # here
                except Exception as e:
                    print(e)
                    continue

                cv2.imshow('lalala', copy_img)

                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
                    continue
                #
                # elif key == ord("n"):
                #     is_skip = True
                #     cv2.destroyAllWindows()
                #     continue
                #
                # elif key == ord("q"):
                #     is_quit = is_quit
                #     cv2.destroyAllWindows()
                #     continue


if __name__ == "__main__":
    GeneralUtils.generate_coco_subset()
